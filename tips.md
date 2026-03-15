1、critic和actor的学习率(微分里的dx)由各自参数量级决定，对每个网络，可以选取为【参数/梯度】的最小量级*1%（每次更新原参数大小的1%）
2、如果不对齐critic adv和actor adv的数量级，开始时adv下降同时ret增加，因为尽管actor在使ret增大，
还是赶不上critic对齐trace return真值（可>1）的上升速度，后来critic饱和不变后，actor才会真正开始后来居上，
属于critic的adv迅速上升后带动actor稳定上升。而如果对齐critic adv和actor adv的数量级，就是两者均在缓慢上升。
3、确保policy输出的actions数量级在个位（符合弧度制，以防pd控制的平衡位置超过限位）
4、提前终止惩罚会被batch平摊，设置成较大值来鼓励更大的trace length
5、随机化时钟来学习到全过程特征
6、确保reward一个量级
序列越长trace ret一定越大，但不代表策略最好（如3帧完成的动作拖到30帧），最好的策略要使平均每时刻的ret最大
网络某些参数训练饱和后grad会显示为0，也可能是梯度消失