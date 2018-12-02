import numpy as np

### Discarded vectorization, cause its p in the a.
### First step at loop implementation, now runs over the input array diagonal-wise



rho1 = np.arange(12).reshape(3,4)
rho2 = np.arange(12).reshape(4,3)
rho3 = np.arange(8).reshape(4,2)
rho4 = np.arange(8).reshape(2,4)


def compute_phi_2d_gauss_seidel(rho, l, max_it=1E5, epsilon_0=1):
    ### define relevant variables ###
    nx = rho.shape[1]
    ny = rho.shape[0]
    n_min = min([nx, ny])
    hx = (nx / l[0]) ** 2
    hy = (ny / l[1]) ** 2
    hxhy = 1 / (2 * hx + 2 * hy)
    #phi = np.zeros((nx, ny))
    phi = rho
    print(rho)

    ### define indexing lists for loop ###
    index_y = [ny - i if ny - i > 0 else 0 for i in range(1, nx + ny)]
    index_x = [i - ny if i - ny > 0 else 0 for i in range(1, nx + ny)]
    indices = [(a,b) for a,b in zip(index_y,index_x)]
    diag_len = [n_min for i in np.ones(nx + ny - 1, dtype=np.int8)]
    for i in range(n_min):
        diag_len[i] = i + 1
        diag_len[-i - 1] = i + 1

    ### run main loop ###
    for k, l in enumerate(indices):
        for m in range(diag_len[k]):
            i = l[0] + m
            j = l[1] + m
            print(rho[i,j])


compute_phi_2d_gauss_seidel(rho1, [1,1])
compute_phi_2d_gauss_seidel(rho2, [1,1])
compute_phi_2d_gauss_seidel(rho3, [1,1])
compute_phi_2d_gauss_seidel(rho4, [1,1])
