import tensorflow as tf
from jax.experimental import jax2tf
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# JAX-based contact geometry functions
# ============================================================================

def get_strain_curvature2D_jax(node0, node1, node2):
    """Compute curvature of 3-node beam segment"""
    node0 = jnp.array([*node0, 0.0])
    node1 = jnp.array([*node1, 0.0])
    node2 = jnp.array([*node2, 0.0])

    ee = node1 - node0
    ef = node2 - node1
    te = ee / jnp.linalg.norm(ee)
    tf = ef / jnp.linalg.norm(ef)
    kb = 2.0 * jnp.cross(te, tf) / (1.0 + jnp.dot(te, tf))
    return kb[2]

def signedAngle_jax(u, v, n):
    """Compute signed angle between vectors u and v"""
    w = jnp.cross(u, v)
    angle = jnp.arctan2(jnp.linalg.norm(w), jnp.dot(u, v))
    angle = jnp.where(jnp.dot(n, w) < 0, -angle, angle)
    return angle

def get_circle_centers_from_q_jax(q):
    """Get osculating circle centers from contact parameters q"""
    x, y, psi, kappa1, kappa2 = q

    R1 = 1.0 / kappa1
    C1_local = jnp.array([-0.5, R1])

    R2 = 1.0 / kappa2
    C2_local_B = jnp.array([-0.5, R2])
    b1_local = jnp.array([x, y])

    R_psi = jnp.array([
        [jnp.cos(psi), -jnp.sin(psi)],
        [jnp.sin(psi),  jnp.cos(psi)]
    ])

    C2_local = R_psi @ C2_local_B + b1_local
    return C1_local, R1, C2_local, R2

def extract_contact_parameters_jax(nodes):
    """Extract 5D contact parameters from 6 nodes"""
    a0, a1, a2, b0, b1, b2 = [nodes[i] for i in range(6)]

    # Compute curvatures
    kappa1 = get_strain_curvature2D_jax(a0, a1, a2)
    kappa2 = get_strain_curvature2D_jax(b0, b1, b2)

    # Rotation matrix R (Beam A frame to global)
    t = a1 - a0
    t_norm = jnp.linalg.norm(t)
    t_unit = t / t_norm
    R = jnp.array([
        [t_unit[0], -t_unit[1]],
        [t_unit[1],  t_unit[0]]
    ])

    # Transform Beam B into Beam A's local frame
    R_T = R.T
    b0_local = R_T @ (b0 - a1)
    b1_local = R_T @ (b1 - a1)
    b2_local = R_T @ (b2 - a1)

    # Extract position and angle
    x, y = b1_local
    e = b1_local - b0_local
    u = jnp.array([1.0, 0.0, 0.0])
    v = jnp.array([e[0], e[1], 0.0])
    n = jnp.array([0.0, 0.0, 1.0])
    psi = signedAngle_jax(u, v, n)

    q = jnp.array([x, y, psi, kappa1, kappa2])
    return q, R

def compute_contact_geometry_jax(q):
    """Compute minimum distance between beam arcs"""
    x, y, psi, kappa1, kappa2 = q
    C1, R1, C2, R2 = get_circle_centers_from_q_jax(q)

    # Beam A: sample arc
    phi1 = 2.0 * jnp.arctan(1.0 / (2.0 * R1))
    YA = signedAngle_jax(jnp.array([1.0,0.0,0.0]), jnp.array([-C1[0], -C1[1], 0.0]), jnp.array([0.0,0.0,1.0]))
    thetaA = jnp.linspace(YA - phi1, YA + phi1, 300)
    arcA = C1 + jnp.abs(R1) * jnp.stack([jnp.cos(thetaA), jnp.sin(thetaA)], axis=1)

    # Beam B: sample arc
    phi2 = 2.0 * jnp.arctan(1.0 / (2.0 * R2))
    YB = signedAngle_jax(jnp.array([1.0,0.0,0.0]), jnp.array([x - C2[0], y - C2[1], 0.0]), jnp.array([0.0,0.0,1.0]))
    thetaB = jnp.linspace(YB - phi2, YB + phi2, 300)
    arcB = C2 + jnp.abs(R2) * jnp.stack([jnp.cos(thetaB), jnp.sin(thetaB)], axis=1)

    # Compute distances
    diffs = arcA[:, None, :] - arcB[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=2)
    idx = jnp.unravel_index(jnp.argmin(dists), dists.shape)
    d_min = dists[idx]

    # Arclengths
    sA = R1 * (thetaA[idx[0]] - YA)
    sB = R2 * (thetaB[idx[1]] - YB)
    sA = sA / (R1 * phi1)
    sB = sB / (R2 * phi2)
    
    return d_min, sA, sB, arcA, arcB

# ============================================================================
# Auto-diff interface
# ============================================================================

@jax.jit
def dmin_jax(x):
    """Main function: compute minimum distance from 12D input"""
    nodes = x.reshape((6,2))
    q, _ = extract_contact_parameters_jax(nodes)
    d_min, *_ = compute_contact_geometry_jax(q)
    return d_min

@jax.jit
def grad_and_hess(x):
    """Compute gradient and Hessian of d_min w.r.t. 12D input"""
    g = jax.grad(dmin_jax)(x)
    H = jax.jacrev(jax.grad(dmin_jax))(x)
    return g, H

# ============================================================================
# Validation functions
# ============================================================================

def compute_norm_err(A_jax, A_fd):
    avg = np.mean(np.abs(A_fd))
    if avg == 0:
        return np.zeros_like(A_fd)
    return np.abs(A_jax - A_fd) / avg

def validate_single():
    """Validate on single random configuration"""
    x = np.random.uniform(-2.0, 2.0, size=(12,)).astype(np.float64)
    
    # Analytical results via JAX
    g_jax, H_jax = grad_and_hess(x)
    
    # Finite-difference approximations
    eps = 1e-4
    g_fd = np.zeros_like(g_jax)
    for i in range(12):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps; xm[i] -= eps
        g_fd[i] = (dmin_jax(xp) - dmin_jax(xm)) / (2*eps)
    
    H_fd = np.zeros_like(H_jax)
    for j in range(12):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps; xm[j] -= eps
        gp, _ = grad_and_hess(xp)
        gm, _ = grad_and_hess(xm)
        H_fd[:,j] = (gp - gm) / (2*eps)

    Lg = compute_norm_err(g_jax, g_fd)
    LH = compute_norm_err(H_jax, H_fd)

    print("=== validate_single ===")
    print(f"Max norm grad error: {np.max(Lg):.2e}")
    print(f"Mean norm grad error:{np.mean(Lg):.2e}")
    print(f"Max norm Hess error: {np.max(LH):.2e}")
    print(f"Mean norm Hess error:{np.mean(LH):.2e}\n")

    # Plot comparison
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(g_jax, 'ro-', label='Analytical (JAX)')
    plt.plot(g_fd,  'b^--',label='Finite-Diff')
    plt.title('Gradient Comparison')
    plt.xlabel('Index'); plt.ylabel('Value')
    plt.legend(); plt.grid()

    plt.subplot(1,2,2)
    plt.plot(H_jax.flatten(), 'ro', label='Analytical (JAX)')
    plt.plot(H_fd.flatten(),  'b^', label='Finite-Diff')
    plt.title('Hessian Comparison')
    plt.xlabel('Flattened Index'); plt.ylabel('Value')
    plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig('single_compare.png')
    plt.show()

def validate_stats(eps=1e-4, trials=1000):
    """Statistical validation over multiple random configs"""
    # Warm-up JIT
    grad_and_hess(np.zeros(12, dtype=np.float64))

    norm_g_errs, norm_H_errs = [], []
    
    for _ in range(trials):
        x0 = np.random.randn(12).astype(np.float64)
        g_jax, H_jax = grad_and_hess(x0)
        
        # Central difference for gradient
        g_fd = np.zeros(12)
        for i in range(12):
            xp = x0.copy(); xm = x0.copy()
            xp[i] += eps; xm[i] -= eps
            g_fd[i] = (dmin_jax(xp) - dmin_jax(xm)) / (2*eps)
            
        # Central difference for Hessian
        H_fd = np.zeros((12,12))
        for j in range(12):
            xp = x0.copy(); xm = x0.copy()
            xp[j] += eps; xm[j] -= eps
            gp, _ = grad_and_hess(xp)
            gm, _ = grad_and_hess(xm)
            H_fd[:,j] = (gp - gm) / (2*eps)
            
        norm_g_errs.append(np.max(compute_norm_err(g_jax, g_fd)))
        norm_H_errs.append(np.max(compute_norm_err(H_jax, H_fd)))
    
    norm_g_errs = np.array(norm_g_errs)
    norm_H_errs = np.array(norm_H_errs)
    
    def print_stats(name, data):
        print(f"{name} over {trials} trials:")
        print(f"  mean  = {data.mean():.2e}")
        print(f"  std   = {data.std():.2e}")
        for p in (50,75,90,95,99):
            print(f"  {p}th pct = {np.percentile(data, p):.2e}")
        print()

    print_stats("Norm-Grad Error", norm_g_errs)
    print_stats("Norm-Hess Error", norm_H_errs)

    # Log-scale histograms
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.hist(norm_g_errs, bins=np.logspace(np.log10(eps/10), np.log10(norm_g_errs.max()*1.1), 20), edgecolor='k')
    plt.xscale('log')
    plt.title('Gradient error (log scale)')
    plt.xlabel('norm_g_errs')
    plt.ylabel('count')
    
    plt.subplot(1,2,2)
    plt.hist(norm_H_errs, bins=np.logspace(np.log10(eps/10), np.log10(norm_H_errs.max()*1.1), 20), edgecolor='k')
    plt.xscale('log')
    plt.title('Hessian error (log scale)')
    plt.xlabel('norm_H_errs')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig('err_hist_log.png')
    plt.close()
    print('Log-scale histogram saved: err_hist_log.png')

# ============================================================================
# TensorFlow export
# ============================================================================

tf_grad_hess = tf.function(
    jax2tf.convert(grad_and_hess, with_gradient=False),
    input_signature=[tf.TensorSpec([12], tf.float64)]
)

class GradHessModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.grad_hess = tf_grad_hess

if __name__ == '__main__':
    validate_single()
    # validate_stats()

    # Optional: save specific test case
    x = np.loadtxt('x.txt').astype(np.float64)
    g_py, H_py = grad_and_hess(x)
    np.savetxt('py_grad.txt', g_py, fmt='%12.6f')
    np.savetxt('py_hess.txt', H_py, fmt='%12.6f', delimiter=' ')
    print('Input x.txt:\nPython grad/hess saved to py_grad.txt & py_hess.txt\n')

    module = GradHessModule()
    tf.saved_model.save(
        module,
        export_dir='exported_model',
        signatures={'serving_default': module.grad_hess}
    )
    print('SavedModel exported to ./exported_model')