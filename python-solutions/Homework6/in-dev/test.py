import trimesh


if __name__ =="__main__":
    filepath = "e:/projects/games101/Homework6/task/Assignment6/models/bunny/bunny.obj"
    mesh = trimesh.load_mesh(filepath)
    verts, faces = mesh.vertices, mesh.faces

    print(verts)


    sdf = mesh.nearest.signed_distance([[-0,  0.1,  0.02]])
    print(sdf)