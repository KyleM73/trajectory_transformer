import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def rk4(func,t,x,u,dt,*args):
    k1 = func(t,x,u,*args)
    k2 = func(t,x+dt*k1/2,u,*args)
    k3 = func(t,x+dt*k2/2,u,*args)
    k4 = func(t,x+dt*k3,u,*args)
    return x+(k1+2*k2+2*k3+k4)*dt/6

cfg = {
    "m1" : 1,
    "l1" : 1,
    "m2" : 1,
    "l2" : 1,
    "g" : 9.8,
    "u_max" : 10,
    "dt": 0.001,
    "horizon" : 10
}

class DoublePendulum(gym.Env):
    def __init__(self,record=False,cfg=cfg,linearize=False):
        self.cfg = cfg
        self.m1 = self.cfg["m1"]
        self.l1 = self.cfg["l1"]
        self.m2 = self.cfg["m2"]
        self.l2 = self.cfg["l2"]
        self.l = self.l1 + self.l2
        self.g = self.cfg["g"]
        self.u_max = self.cfg["u_max"]
        self.dt = self.cfg["dt"]
        self.T = int(self.cfg["horizon"] / self.dt)
        self.record = record
        self.linearize = linearize

        if self.linearize:
            self.x_star = np.array([np.pi,0,0,0]).reshape((-1,1))
            self.u_star = np.array([0,0]).reshape((-1,1))
            self.Alin,self.Blin = self.get_lin_dyn(self.x_star,self.u_star)

        self.observation_space = gym.spaces.Box(
            low=np.array([np.float64(n) for n in [-np.pi,-np.pi,-np.inf,-np.inf]]),
            high=np.array([np.float64(n) for n in [np.pi,np.pi,np.inf,np.inf]]),
            shape=(4,),dtype=np.float64)
        self.action_space = gym.spaces.Box(
            low=np.array([np.float64(n) for n in [-self.u_max,-self.u_max]]),
            high=np.array([np.float64(n) for n in [self.u_max,self.u_max]]),
            shape=(2,),dtype=np.float64)

    def reset(self):
        if self.linearize:
            rng = np.random.default_rng()
            self.x = self.x_star + 0.1*rng.random((self.observation_space.shape[0],1))-0.05
            self.u = self.u_star
        else:
            self.x = self.observation_space.sample()
            #self.x[2:] = 0
            self.u = np.zeros(self.action_space.sample().shape)
        self.q0 = self.q = self.x[:2]
        self.qdot0 = self.qdot = self.x[2:]
        if self.record: self.init_hist()
        return self.x

    def init_hist(self):
        self.x_hist = self.x.reshape((1,-1))
        self.u_hist = self.u.reshape((1,-1))

    def step(self,action):
        self.u = np.clip(action,-self.u_max,self.u_max)
        self.dynamics(self.u)
        if self.record: self.save_hist()
        return self.x,0,False,dict()

    def dynamics(self,u=0):    
        if isinstance(u,np.ndarray):
            u = u.reshape((-1,1))
        else:
            u = np.array(u).reshape((-1,1))
        def dyn(t,x,u):
            M = np.array([
                [(self.m1 + self.m2)*self.l1**2 + self.m2*self.l2**2 + 2*self.m2*self.l1*self.l2*np.cos(x[1]), self.m2*self.l2**2 + self.m2*self.l1*self.l2*np.cos(x[1])],
                [self.m2*self.l2**2 + self.m2*self.l1*self.l2*np.cos(x[1]), self.m2*self.l2**2]
                ])
            C = np.array([
                [0, -self.m2*self.l1*self.l2*(2*x[2] + x[3])*np.sin(x[1])],
                [0.5*self.m2*self.l1*self.l2*(2*x[2] + x[3])*np.sin(x[1]), -0.5*self.m2*self.l1*self.l2*x[2]*np.sin(x[1])]
                ])
            Tg = -self.g*np.array([
                [(self.m1 + self.m2)*self.l1*np.sin(x[0]) + self.m2*self.l2*np.sin(x[0]+x[1])],
                [self.m2*self.l2*np.sin(x[0]+x[1])]
                ])
            B = np.array([[1,0],[0,1]])
            qddot = np.linalg.inv(M) @ ( Tg + B @ u - C @ x[2:].reshape((-1,1)) )
            xdot = np.hstack((x[2:],qddot.reshape((-1,))))
            return xdot
        def lin_dyn(t,x,u,x_star,u_star):
            xbar = x-x_star
            ubar = u-u_star
            return self.Alin@xbar + self.Blin@ubar
        if self.linearize:
            self.x = rk4(lin_dyn,0,self.x,u,self.dt,self.x_star,self.u_star)
        else:
            self.x = rk4(dyn,0,self.x,u,self.dt)
        self.x = self.wrap(self.x)
        self.q = self.x[:2]
        self.qdot = self.x[2:]

    def get_lin_dyn(self,x_star,u_star):
        M = np.array([
            [(self.m1 + self.m2)*self.l1**2 + self.m2*self.l2**2 + 2*self.m2*self.l1*self.l2*np.cos(x_star[1,0]), self.m2*self.l2**2 + self.m2*self.l1*self.l2*np.cos(x_star[1,0])],
            [self.m2*self.l2**2 + self.m2*self.l1*self.l2*np.cos(x_star[1,0]), self.m2*self.l2**2]
            ])
        dTdq = -self.g*np.array([
            [(self.m1 + self.m2)*self.l1 + self.m2*self.l2, self.m2*self.l2],
            [self.m2*self.l2, self.m2*self.l2]
            ])
        Alin = np.vstack((
            np.hstack(( np.zeros((2,2)), np.identity(2) )),
            np.hstack(( np.linalg.inv(M)@dTdq, np.zeros((2,2)) )) 
            ))
        B = np.array([[1,0],[0,1]])
        Blin = np.vstack((np.zeros((2,2)),np.linalg.inv(M)@B))
        return Alin,Blin

    def save_hist(self):
        self.x_hist = np.vstack((self.x_hist,self.x.reshape((1,-1))))
        self.u_hist = np.vstack((self.u_hist,self.u.reshape((1,-1))))
        
    def get_hist(self):
        return self.x_hist,self.u_hist
        
    def render(self):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(autoscale_on=False, xlim=(-self.l-0.1,self.l+0.1),ylim=(-self.l-0.1,self.l+0.1))
        self.ax.set_aspect('equal')
        self.ax.grid()
        plt.xticks([]) 
        plt.yticks([]) 
        title_str = "q = "+str(self.q0.reshape((-1,)))+" qdot = "+str(self.qdot0.reshape((-1,)))
        self.ax.set_title(title_str, fontsize=16)

        self.line1, = self.ax.plot([], [], 'o-', lw=2)
        self.line2, = self.ax.plot([], [], 'o-', lw=2)
        self.trace, = self.ax.plot([], [], '.-', lw=1, ms=2)
        self.time_template = 'time = %.1fs'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

        x1 = self.l1 * np.sin(self.x_hist[:,0])
        y1 = -self.l1 * np.cos(self.x_hist[:,0])
        x2 = x1 + self.l2 * np.sin(self.x_hist[:,0]+self.x_hist[:,1])
        y2 = y1 - self.l2 * np.cos(self.x_hist[:,0]+self.x_hist[:,1])

        def animate(i):
            scale = int(0.01/self.dt)
            self.line1.set_data([0,x1[i]],[0,y1[i]])
            self.line2.set_data([x1[i],x2[i]],[y1[i],y2[i]])
            if i > 50*scale:
                self.trace.set_data(x2[i-50*scale:i:scale],y2[i-50*scale:i:scale])
            else:
                self.trace.set_data(x2[:i:scale],y2[:i:scale])
            self.time_text.set_text(self.time_template % (i*self.dt))
            return self.line1, self.line2, self.trace, self.time_text

        self.fig.tight_layout()
        self.ani = animation.FuncAnimation(self.fig, animate, range(0,len(y2),int(0.01/self.dt)), interval=10, blit=True)

        t = np.array([i*self.dt for i in range(self.T)])
        
        self.fig_x,self.ax_x = plt.subplots(figsize=(8, 6))
        self.ax_x.plot(t,self.x_hist[:,0],label="q1")
        self.ax_x.plot(t,self.x_hist[:,1],label="q2")
        self.ax_x.plot(t,self.x_hist[:,2],label="q1dot")
        self.ax_x.plot(t,self.x_hist[:,3],label="q2dot")
        self.ax_x.plot(t,self.u_hist[:,0],label="u1")
        self.ax_x.plot(t,self.u_hist[:,1],label="u2")
        self.ax_x.legend()
        self.fig_x.tight_layout()
        plt.show()

    def close(self):
        if self.record: self.render()

    def wrap(self,x):
        q = x[:2]
        q = q%(2*np.pi)
        q = (q+2*np.pi)%(2*np.pi)
        q = np.array([qq - 2*np.pi if qq > np.pi else qq for qq in q]).reshape((-1,))
        x[:2] = q
        return x

if __name__ == "__main__":
    env = DoublePendulum(record=True)
    env.reset()
    for i in range(env.T-1):
        action = np.zeros((env.action_space.shape))
        x,_,_,_ = env.step(action)
    env.close()



