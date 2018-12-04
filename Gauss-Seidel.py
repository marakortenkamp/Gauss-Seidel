import numpy as np

### Discarded vectorization, cause its a p in the a.
### implemented loop for 1000 iterations with 4 test charge distribution
### every element phi_ij is computed as follows:
### phi_i,j = 1 / (2/hx**2 + 2/hy**2) * ( (phi_x+1,y + phi_x-1,y) / hx**2
###                                     + (phi_x,y+1 + phi_x,j-1) / hy**2
###                                     + 1/epsilon_0 * rho_x,y
###                                     )

### something is wrong tho, since all elements in phi equal
### feedback appreciated


rho1 = np.zeros((5,5))
rho1[0,0] = 10
rho1[-1,-1] = -10

rho2 = np.zeros((5,6))
rho2[0,0] = 10
rho2[-1,-1] = -10

rho3 = np.zeros((3,6))
rho3[0,0] = 10
rho3[-1,-1] = -10

def compute_phi_2d_gauss_seidel(rho, l, max_it=1E5, epsilon_0=1):
    ### define relevant variables ###
    nx = rho.shape[1]
    ny = rho.shape[0]
    n_min = min([nx, ny])
    hx = (nx / l[0]) ** 2
    hy = (ny / l[1]) ** 2
    hxhy = 1 / (2 * hx + 2 * hy)
    phi = np.zeros((ny, nx))


    ### define indexing lists for loop ###
    index_y  = [ny - i if ny - i > 0 else 0 for i in range(1, nx + ny)]
    index_x  = [i - ny if i - ny > 0 else 0 for i in range(1, nx + ny)]
    indices  = [(a,b) for a,b in zip(index_y,index_x)]
    diag_len = [n_min for i in np.ones(nx + ny - 1, dtype=np.int8)]
    for i in range(n_min):
        diag_len[i] = i + 1
        diag_len[-i-1] = i + 1

    ### run main loop ###
    for n_iteration in range(20000):
        for k, l in enumerate(indices):
            for m in range(diag_len[k]):
                i = l[0] + m
                j = l[1] + m
                phi_up = phi[i-1,j]
                phi_left = phi[i,j-1]
                if i < ny - 1:
                    phi_down = phi[i+1,j]
                else:
                    phi_down = phi[0,j]
                if j < nx - 1:
                    phi_right = phi[i,j+1]
                else:
                    phi_right = phi[i,0]
                phi[i,j] = (hxhy * ( hx * (phi_right + phi_left)
                                   + hy * (phi_up + phi_down)
                                   + 1 / epsilon_0 * rho[i,j]
                                   ))

    return phi


print(rho1)
print(compute_phi_2d_gauss_seidel(rho1, [4, 4]))
print(rho2)
print(compute_phi_2d_gauss_seidel(rho2, [4, 4]))
print(rho3)
print(compute_phi_2d_gauss_seidel(rho3, [4, 4]))
