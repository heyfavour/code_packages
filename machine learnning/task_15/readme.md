
## PG
#### policygradient_reinforce.py
```
loss = -R =  -reward*log(p)
```
![](utils/pictures/PG.png)

### AC:
loss = -R =  -Q(s,a)*log(p) 使用Q函数来代替R
#### Actor-Critic.py
```
loss = actor_loss + critic_loss
loss  =-Q*log(p) + loss_func(value,R) = =-(sum_r-v)*log(p) + loss_func(value,R)
```
#### actor 
![](utils/pictures/AC_actor.png)
#### critic
![](utils/pictures/AC_loss.png)



### A2C
#### A2C.py
#### actor 
![](utils/pictures/A2C_actor.png)
![](utils/pictures/A2C_critic.png)
```
advantage = A = Q-V = r+v_next-v=td error=sum(reward+0.95*v_next*(1-done)) - values
#pg：-reward*log(p) 
actor_loss  = -advantage * log(p) 
```
#### critic
```
critic_loss = advantage.pow(2).mean()
```
#### loss
```
loss=actor_loss + 0.5 * critic_loss - 0.001 * entropy
```
### 参考url
```
https://zhuanlan.zhihu.com/p/51645768

https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
```

