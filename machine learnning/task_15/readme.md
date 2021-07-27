## PG
#### policygradient_reinforce.py
```
loss = -R =  -reward*log(p)
```


### AC:
#### Actor-Critic.py
```
acotr_loss = -R =  -reward*log(p) 
= -(y*r-b)*log(p)
= -(y*r-v)*log(p)
loss = actor_loss + critic_loss
loss =-(y*r-v)*log(p) + loss_func(value,R) 
     =-(y*r-v)*log(p) + loss_func(value,R)
```
### A2C
#### A2C.py
#### actor
```
advantage = A = Q-V = r+v_next-v=td error=sum(reward+0.95*v_next*(1-done)) - values
#pg：-reward*log(p) 
acotr_loss = -R = -reward*log(p) 
= -(y*r-b)*log(p)
= -(r+v_next-v)*log(p)
= - advantage*log(p)

critic_loss = advantage.pow(2).mean()

loss=actor_loss + 0.5 * critic_loss - 0.001 * entropy
```
### 参考url
```
https://zhuanlan.zhihu.com/p/51645768

https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
```

