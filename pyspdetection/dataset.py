import numpy as np

class Dataset():

    def __init__(self, temporal_networks, dynamics):
        """
        temporal_networks: TemporalNetwork instance properly initialized
        dynamics : BaseDynamics instance
        """
        self.networks = temporal_networks
        self.dynamics = dynamics
        return

    def time_series(self, T, burn=0):
        X0 = self.dynamics.generate_x0(self.networks(0))
        return self.get_time_series(X0, T, burn=burn)

    def get_time_series(self, X0, T, burn=0):

        if burn > 0:
            dt = T[1]-T[0]
            t_burn = np.arange(0,burn*dt, dt)
            x = self.dynamics.time_series(X0.copy(), self.networks(0), t_burn)
            X0 = x[-1].copy()

        events = self.networks.perturbations_events.copy()

        X = []
        i0 = 0
        G = self.networks(events[0])
        i0 += 1
        istart = 0
        last_integration = False
        
        while last_integration==False:
            
            if i0>=len(events):
                next_t = T[-1]
                last_integration = True
            else:
                if events[i0] > T[-1]:
                    next_t = T[-1]
                    last_integration = True        
                else:
                    next_t = events[i0]
                    i0 += 1
                    
            i1 = np.argmin(np.abs(T-next_t))+1
            Tsub = T[istart:i1]
            istart = i1
            
            # Change network
            G = self.networks((Tsub[0]+Tsub[-1])/2)
            
            u = self.dynamics.time_series(X0.copy(), G, Tsub)
            X0 = u[-1].copy()
            X.append(u.copy())
        
        return np.concatenate(X, axis=0)