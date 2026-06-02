#!/usr/bin/env python3
import os
import signal
import tempfile
import atexit
import torch
import torch.nn as nn
import numpy as np
import swanlab

class ModelSaver:
    def __init__(self, model, experiment_dir):
        self.model = model
        self.experiment_dir = experiment_dir
        self.should_save = False
        self.save_lock = False
        
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup_save)

    def signal_handler(self, sig, frame):
        print("\n[信号拦截] 捕获系统紧急终止指令，标记中断信号...")
        self.should_save = True

    def save_checkpoint(self, checkpoint_name):
        if self.save_lock:
            return
        try:
            self.save_lock = True
            model_fn = f"{self.experiment_dir}/{checkpoint_name}.pt"
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name
                state_dict = {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}
                torch.save(state_dict, tmp_path)
                os.replace(tmp_path, model_fn)
                print(f"🎉 成功保存模型至: [{checkpoint_name}.pt]")
        except Exception as e:
            print(f"⚠️ 保存中断故障: {str(e)}")
        finally:
            self.save_lock = False
            self.should_save = False

    def cleanup_save(self):
        if not self.should_save:
            self.save_checkpoint("model_interrupted_backup")

# =========================================================================
# 3. 🛡️ 完全看齐 GCBF+ 论文：后台图级运动学前向推演算子
# =========================================================================
def forward_graph_hetero(batch_curr, pred_action, dt=0.05):
    """
    统一异质图级前向动力学演进引擎。
    输入当前大异质图 batch 与网络预测出的实时控制指令，
    自动平移自车节点并完成所有变长雷达碰撞点局部坐标系下的刚体差速逆变换。
    """
    batch_next = batch_curr.clone()
    
    pred_v, pred_w = pred_action[:, 0:1], pred_action[:, 1:2]
    state_curr = batch_curr['ego'].x
    
    x_c, y_c, th_c, dist_c = state_curr[:, 0:1], state_curr[:, 1:2], state_curr[:, 2:3], state_curr[:, 3:4]
    
    # 3.1 演进自车状态 (Kinematic Motion Model)
    x_next = x_c + pred_v * torch.cos(th_c) * dt
    y_next = y_c + pred_v * torch.sin(th_c) * dt
    th_next = th_c + pred_w * dt
    
    # 靶点物理距离项同步积分更新
    dist_next = dist_c - (pred_v * torch.cos(th_c) * (x_c / (dist_c + 1e-6)) + pred_v * torch.sin(th_c) * (y_c / (dist_c + 1e-6))) * dt
    batch_next['ego'].x = torch.cat([x_next, y_next, th_next, dist_next], dim=-1)
    
    # 3.2 演进载体系引力目标节点 (更新自车局部系下的相对目标位置)
    # 利用相同的线速度和角速度，反向推演目标局部位置在下一步的变化
    target_local_x_curr = batch_curr['goal'].x[:, 0:1]
    target_local_y_curr = batch_curr['goal'].x[:, 1:2]
    
    # 刚体反向平移与旋转更新目标相对位置
    delta_th_ego = pred_w * dt
    cos_dth_e = torch.cos(delta_th_ego)
    sin_dth_e = torch.sin(delta_th_ego)
    
    t_x_trans = target_local_x_curr - pred_v * dt
    t_y_trans = target_local_y_curr
    
    target_local_x_next = cos_dth_e * t_x_trans + sin_dth_e * t_y_trans
    target_local_y_next = -sin_dth_e * t_x_trans + cos_dth_e * t_y_trans
    batch_next['goal'].x = torch.cat([target_local_x_next, target_local_y_next], dim=-1)
    
    # 3.3 演进周围异质雷达碰撞点云拓扑 (局部系点阵逆刚体差速平移)
    if batch_curr['point'].x.shape[0] > 0:
        node_batch_idx = batch_curr['point'].batch  # 动态解锁 13015 个雷达点与 256台车的归属关系
        
        # 利用高级索引机制瞬间将 256 维控制动作广播放大至变长点云空间尺度上
        v_per_point = pred_v[node_batch_idx]       
        w_per_point = pred_w[node_batch_idx]       
        
        delta_theta = w_per_point * dt
        cos_dth = torch.cos(delta_theta)
        sin_dth = torch.sin(delta_theta)
        
        pc_x = batch_curr['point'].x[:, 0:1]
        pc_y = batch_curr['point'].x[:, 1:2]
        
        pc_x_trans = pc_x - v_per_point * dt
        pc_y_trans = pc_y
        
        pc_x_next = cos_dth * pc_x_trans + sin_dth * pc_y_trans
        pc_y_next = -sin_dth * pc_x_trans + cos_dth * pc_y_trans
        
        batch_next['point'].x = torch.cat([pc_x_next, pc_y_next], dim=-1)
        batch_next['point'].batch = node_batch_idx
        
    return batch_next

# =========================================================================
# 4. 主训练优化步长演进器
# =========================================================================
def train_epoch(model, train_loader, optimizer, criterion_ctrl, saver, args, scheduler, device, epoch, best_tracker):
    model.train()
    scaler = torch.amp.GradScaler()
    
    total_loss, total_loss_ctrl, total_loss_risk, total_loss_phys = 0.0, 0.0, 0.0, 0.0
    interrupted = False 
    criterion_risk = nn.MSELoss(reduction='none')
    
    # 这里的 batch 是纯正、标准、由 PyG DataLoader 直接打包吐出的三实体联合异质图 Batch 对象
    for batch in train_loader:
        if saver.should_save:
            saver.save_checkpoint(f"epoch_interrupt_step_{epoch}")
            interrupted = True
            break

        # 搬运一整张大图上天线
        batch = batch.to(device, non_blocking=True)
        labels = batch.y                      # 多任务专家复合监督标签 [Batch_Size, 5]
        dist_c = batch['ego'].x[:, 3:4]        # 从自车特征第4维动态解包出目标相对距离 d(x)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # 解包监督标签与专家靶点
            true_action = labels[:, :2]         # 专家控制指令 (v_true, w_true)
            true_psi = labels[:, 2:3]           # 空间标量 psi
            nominal_action = labels[:, 3:5]     # 名义直行控制动作 (v_nom, w_nom)
            risk_target = 1.0 - true_psi        # 解耦安全风险互补面目标
            
            # 🟢 分支 A：计算当前时刻 t 的网络前向控制指令与预测风险
            pred_action, pred_risk_curr = model(batch)
            pred_v, pred_w = pred_action[:, 0:1], pred_action[:, 1:2]
            
            # =========================================================================
            # 🟢【自监督图上演进魔术】：调用纯矩阵化后台推演算子更新未来图结构
            # =========================================================================
            batch_next = forward_graph_hetero(batch, pred_action, dt=0.05)
            dist_next = batch_next['ego'].x[:, 3:4] # 自动提取下一步的更新靶距
            
            # 🟢 分支 B：预测未来虚拟图拓扑下的风险场强值度量
            _, pred_risk_next = model(batch_next)
            
            # =========================================================================
            # 🟢【时序差分 PINN 守恒硬约束】：构建场强物质时间变化率
            # =========================================================================
            safe_dist_curr = torch.clamp(dist_c, min=0.2)
            safe_dist_next = torch.clamp(dist_next, min=0.2)
            alpha = 0.5
            rho_curr = (1.0 - pred_risk_curr) / (safe_dist_curr ** alpha)
            rho_next = (1.0 - pred_risk_next) / (safe_dist_next ** alpha)
            
            # 一阶时序差分逼近时间全导数 \dot{\rho}
            dot_rho_time = (rho_next - rho_curr) / 0.05
            
            # 施加控制论安全守恒流体不等式硬约束 \dot{\rho}_{time} \ge \rho_{curr}
            violation = rho_curr - dot_rho_time
            loss_physics = torch.mean(torch.clamp(violation, min=0.0))
            
            # =========================================================================
            # 多任务损失函数多指标平衡计算
            # =========================================================================
            # --- 任务 1：控制动作仿真误差损失 (加持高价值避障非对称大增益) ---
            loss_ctrl = criterion_ctrl(pred_action, 10*true_action)
            # action_deviation = torch.norm(true_action - nominal_action, dim=1, keepdim=True)
            # action_weight = torch.ones_like(action_deviation)
            # action_weight[action_deviation > 0.05] = 40.0  
            # loss_ctrl = torch.mean(loss_ctrl * action_weight)
            
            # --- 任务 2：几何风险误差损失 (高强度危险边缘加权) ---
            loss_risk_raw = criterion_risk(pred_risk_curr, risk_target)
            risk_weight = torch.ones_like(risk_target)
            risk_weight[risk_target > 0.2] = 10.0
            loss_risk = torch.mean(loss_risk_raw * risk_weight)

            goal_local_pos = batch['goal'].x  
            target_local_x = goal_local_pos[:, 0:1]
            target_local_y = goal_local_pos[:, 1:2]
            
            # 2. 动态审计当前的相对距离
            goal_dist_sq = target_local_x**2 + target_local_y**2 + 1e-6
            goal_dist = torch.sqrt(goal_dist_sq)
            
            # 3. 施加硬核目标控制论状态缩闭环：
            # 物理含义：如果距离目标极近（如 < 0.25m），网络预测的线速度 v 和角速度 w 必须平稳归零摆停！
            # 利用平滑的连续边界权重激活：当距离大于 0.25m 时权重为 0，进入 0.25m 内时权重随逼近呈指数级暴涨
            at_goal_mask = torch.where(goal_dist < 0.25, 1.0, 0.0)
            
            loss_goal_v = torch.mean((pred_v - 0.0)**2 * at_goal_mask)
            loss_goal_w = torch.mean((pred_w - 0.0)**2 * at_goal_mask)
            loss_goal = loss_goal_v + loss_goal_w * 2.0
            
            # 跨模态联合总全优化损失目标
            loss = loss_ctrl
        
        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        total_loss_ctrl += loss_ctrl.item()
        total_loss_risk += loss_risk.item()
        total_loss_phys += loss_physics.item()

    if interrupted:
        return True

    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_loss_ctrl = total_loss_ctrl / num_batches
    avg_loss_risk = total_loss_risk / num_batches
    avg_loss_phys = total_loss_phys / num_batches
    lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1}/{args.num_epochs}] | Combined: {avg_loss:.4f} | "
          f"Ctrl: {avg_loss_ctrl:.4f} | Risk: {avg_loss_risk:.4f} | Phys_TD: {avg_loss_phys:.4f}")

    if args.swanlab:
        swanlab.log({
            "Loss/Total_Combined": avg_loss,
            "Loss/Action_Imitation_Huber": avg_loss_ctrl,
            "Loss/Safety_Complementary_MSE": avg_loss_risk,
            "Loss/Physics_TD_Barrier": avg_loss_phys,
            "Optimization/Adaptive_Learning_Rate": lr
        }, step=epoch)

    time_to_save = (epoch + 1) % args.num_epoch_save == 0
    if time_to_save and avg_loss_risk < best_tracker["best_risk_loss"]:
        best_tracker["best_risk_loss"] = avg_loss_risk
        saver.save_checkpoint("model_best")
        
    return False