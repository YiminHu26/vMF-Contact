import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Generate two random points **inside** the sphere
def generate_random_points_in_sphere(S, R_s, num_points=10):
    """
    Generates random points uniformly distributed on the sphere surface.
    """
    theta = np.random.uniform(0, np.pi/2, num_points)  # Polar angle
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # Azimuthal angle
    R_s = np.random.uniform(0, 1, num_points)*R_s  # Random radius

    X = S[0] + R_s * np.sin(theta) * np.cos(phi)
    Y = S[1] + R_s * np.sin(theta) * np.sin(phi)
    Z = S[2] + R_s * np.cos(theta)

    sphere_points = np.vstack((X, Y, Z)).T
    return sphere_points

    
def generate_random_points_on_sphere(S, R_s, num_points=10):
    """
    Generates uniform points uniformly distributed on the sphere surface.
    """
    theta = np.linspace(0, np.pi, num_points)  # Polar angle
    phi = np.linspace(0, 2 * np.pi, num_points)  # Azimuthal angle

    sphere_points = []
    for t in theta:
        for p in phi:
            X = S[0] + R_s * np.sin(t) * np.cos(p)
            Y = S[1] + R_s * np.sin(t) * np.sin(p)
            Z = S[2] + R_s * np.cos(t)
            sphere_points.append([X, Y, Z])

    return np.array(sphere_points)

# 2. Compute the **intersection circle** between the plane and sphere
def compute_fixed_intersection_circle(S, R_s, P1, P2):
    """
    Computes the intersection circle between the sphere and the perpendicular plane at the midpoint of P1 and P2.
    The plane does NOT pass through the sphere center but through the midpoint of P1 and P2.
    """
    # Compute the midpoint (this is where the plane passes through)
    C = (P1 + P2) / 2

    # Compute the normal of the plane (vector from P1 to P2)
    N = P2 - P1
    N = N / np.linalg.norm(N)

    # Compute the distance from the sphere center to the plane
    d = abs(np.dot(S - C, N))

    # Ensure the plane intersects the sphere
    if d >= R_s:
        raise ValueError("The plane does not intersect the sphere (d >= R_s)")

    # Compute the correct radius of the intersection circle
    R_c = np.sqrt(R_s**2 - d**2)

    # Compute the projection of the sphere center onto the plane (correct circle center)
    C_circle = S - np.dot((S - C), N) * N

    return C, C_circle, R_c, N

# 4. Query **tangent vector at a specific point**
def query_tangent_vector(S, R_s, P1, P2, P_q):
    """
    Queries the tangent vector at a given point on the sphere.
    """
    # Ensure P_q is on the sphere (project if necessary)
    P_q = S + R_s * (P_q - S) / np.linalg.norm(P_q - S)

    # Compute the normal of the plane (vector from P1 to P2)
    N = P2 - P1
    N = N / np.linalg.norm(N)

    # Compute radial vector at P_q
    radial_vector = P_q - S
    radial_vector = radial_vector / np.linalg.norm(radial_vector)

    # Compute rejection of N onto the radial vector
    proj_N_on_radial = np.dot(N, radial_vector) * radial_vector
    tangent_vector = N - proj_N_on_radial  # Remove radial component

    # Normalize to maintain consistent vector lengths
    tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)

    # Compute the projection of S onto the plane perpendicular to P1 P2
    P_mid = (P1 + P2) / 2
    dist_S_plane = np.dot(P_mid - S, N)
    alpha = (dist_S_plane + R_s) / R_s 

    # Get the starting pole of the field
    P_pole = S - R_s * N

    # Compute the strength of the field
    theta = np.arccos(np.dot(P_pole - S, P_q - S) / R_s**2) / np.pi # normalize to [0, 1]
    strength = 1 - theta**alpha
    return tangent_vector * strength

# 5. Plot the **sphere, intersection circle, and tangent vector field**
def plot_query_tangent_vectors(S, R_s, num_queries=20, Ps_num=3):
    """
    Plots a 3D sphere and multiple query points with their tangent vectors.
    """
    # Generate P1 and P2 inside the sphere
    Ps = generate_random_points_in_sphere(S, R_s, Ps_num)

    # Generate multiple query points on the sphere
    query_points = generate_random_points_on_sphere(S, R_s, num_queries)
    
    # Compute tangent vectors at each query point
    P0 =  Ps[1] + (Ps[2] - Ps[1]) * 0.4 # Ps[0] #  
    tangent_vectors = 0
    for P in Ps[1:]:
        tangent_vectors += np.array([query_tangent_vector(S, R_s, P, P0, P_q) for P_q in query_points])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere surface
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    X = S[0] + R_s * np.outer(np.sin(u), np.cos(v))
    Y = S[1] + R_s * np.outer(np.sin(u), np.sin(v))
    Z = S[2] + R_s * np.outer(np.cos(u), np.ones_like(v))
    ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.3)

    # Plot P1, P2, and the query points
    ax.scatter(*P0, color='green', label="P1", s=200)
    for i, P in enumerate(Ps[1:]):
        ax.scatter(*P, color='blue', label=f"P{i+2}", s=200)
    ax.scatter(*np.array(query_points).T, color='purple', label="Query Points", s=20)

    # Plot tangent vectors
    query_points = np.array(query_points)
    ax.quiver(query_points[:, 0], query_points[:, 1], query_points[:, 2],
              tangent_vectors[:, 0], tangent_vectors[:, 1], tangent_vectors[:, 2],
              length=1, color='red', label="Tangent Vectors")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Query Tangent Vectors on the Sphere")

    # Ensure equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])
    max_range = R_s * 1.2
    ax.set_xlim(S[0] - max_range, S[0] + max_range)
    ax.set_ylim(S[1] - max_range, S[1] + max_range)
    ax.set_zlim(S[2] - max_range, S[2] + max_range)

    plt.show()


if __name__ == "__main__":
    # Example Usage
    S = np.array([0, 0, 0])  # Sphere Center
    R_s = 5  # Sphere Radius

    # Plot the results
    plot_query_tangent_vectors(S, R_s)
