def _create_topple_graph(cvh_mesh, com):
    """
    Constructs a toppling digraph for the given convex hull mesh and
    center of mass.

    Each node n_i in the digraph corresponds to a face f_i of the mesh and is
    labelled with the probability that the mesh will land on f_i if dropped
    randomly. Not all faces are stable, and node n_i has a directed edge to
    node n_j if the object will quasi-statically topple from f_i to f_j if it
    lands on f_i initially.

    This computation is described in detail in
    http://goldberg.berkeley.edu/pubs/eps.pdf.

    Parameters
    ----------
    cvh_mesh : trimesh.Trimesh
      Rhe convex hull of the target shape
    com : (3,) float
      The 3D location of the target shape's center of mass

    Returns
    -------
    graph : networkx.DiGraph
      Graph representing static probabilities and toppling
      order for the convex hull
    """
    adj_graph = nx.Graph()
    topple_graph = nx.DiGraph()

    # Create face adjacency graph
    face_pairs = cvh_mesh.face_adjacency
    edges = cvh_mesh.face_adjacency_edges

    graph_edges = []
    for fp, e in zip(face_pairs, edges):
        verts = cvh_mesh.vertices[e]
        graph_edges.append([fp[0], fp[1], {'verts': verts}])

    adj_graph.add_edges_from(graph_edges)

    # Compute static probabilities of landing on each face
    for i, tri in enumerate(cvh_mesh.triangles):
        prob = _compute_static_prob(tri, com)
        topple_graph.add_node(i, prob=prob)

    # Compute COM projections onto planes of each triangle in cvh_mesh
    proj_dists = np.einsum('ij,ij->i', cvh_mesh.face_normals,
                           com - cvh_mesh.triangles[:, 0])
    proj_coms = com - np.einsum('i,ij->ij', proj_dists, cvh_mesh.face_normals)
    barys = points_to_barycentric(cvh_mesh.triangles, proj_coms)
    unstable_face_indices = np.where(np.any(barys < 0, axis=1))[0]

    # For each unstable face, compute the face it topples to
    for fi in unstable_face_indices:
        proj_com = proj_coms[fi]
        centroid = cvh_mesh.triangles_center[fi]
        norm = cvh_mesh.face_normals[fi]

        for tfi in adj_graph[fi]:
            v1, v2 = adj_graph[fi][tfi]['verts']
            if np.dot(np.cross(v1 - centroid, v2 - centroid), norm) < 0:
                tmp = v2
                v2 = v1
                v1 = tmp
            plane1 = [centroid, v1, v1 + norm]
            plane2 = [centroid, v2 + norm, v2]
            if _orient3dfast(plane1, proj_com) >= 0 and _orient3dfast(
                    plane2, proj_com) >= 0:
                break

        topple_graph.add_edge(fi, tfi)

    return topple_graph