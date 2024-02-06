import numpy as np
from scipy.spatial.transform import Rotation as R
from mayavi import mlab
import time
import os


def sort_array_by_columns(arr):
    for i in range(3):
        arr = arr[arr[:,i].argsort()]
    return arr


def min_trans_to_first_quadrant(arr):
    min_vals = np.amin(arr, 0)
    return arr - np.full(arr.shape, min_vals)


first_rotations = [
    R.from_rotvec([0, 0, phi])
    for phi in np.linspace(0, 2 * np.pi, 4, endpoint=False)
]

second_rotations = [
    R.from_rotvec([0, phi, 0])
    for phi in np.linspace(0, 2 * np.pi, 4, endpoint=False)
]
second_rotations += [
    R.from_rotvec([-np.pi/2, 0, 0]),
    R.from_rotvec([np.pi/2, 0, 0])
]


def duplicate_in_solutions(cuboid, dimensions, found_solutions):
    if dimensions not in found_solutions.keys():
        return False

    for rot1 in first_rotations:
        rot1_cuboid = np.array([np.rint(rot1.apply(cube)) for cube in cuboid])
        for rot2 in second_rotations:
            rot2_cuboid = np.array([np.rint(rot2.apply(cube)) for cube in rot1_cuboid])
            rot2_cuboid = min_trans_to_first_quadrant(rot2_cuboid)
            rot2_cuboid = sort_array_by_columns(rot2_cuboid)
            for found_cuboid in found_solutions[dimensions]:
                if np.array_equal(rot2_cuboid, found_cuboid):
                    return True
    return False


def make_cuboid(cube, cuboid, direc, found_solutions):
    new_cube = cube + direc
    for old_cube in cuboid:
        if np.array_equal(new_cube, old_cube):
            return None, None
    new_candidate = np.copy(cuboid)
    new_candidate = np.append(new_candidate, np.array([new_cube]), axis=0)
    new_candidate = min_trans_to_first_quadrant(new_candidate)
    new_candidate = sort_array_by_columns(new_candidate)
    dimensions = tuple(np.sort(np.amax(new_candidate, 0)))
    if duplicate_in_solutions(new_candidate, dimensions, found_solutions):
        return None, None
    return dimensions, new_candidate


directions = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1]),
    np.array([-1, 0, 0]),
    np.array([0, -1, 0]),
    np.array([0, 0, -1]),
]


solutions = [{(0, 0 , 0): [np.array([[0, 0, 0]])]}]

print("size of cuboids:")
n = int(input())

stat_file = open('cuboid_stats.tsv', 'w')

stat_file.write('n\tsolutions\ttime[s]\n')
stat_file.write(f'1\t{len(solutions[-1])}\t0\n')
for i in range(1, n):
    t_start = time.perf_counter()
    solutions.append(dict())
    for cuboid in np.concatenate(list(solutions[-2].values())):
        for cube in cuboid:
            for direc in directions:
                key, new_cuboid = make_cuboid(cube, cuboid, direc, solutions[-1])
                if key == None: continue
                if key not in solutions[-1].keys():
                    solutions[-1][key] = []
                solutions[-1][key].append(new_cuboid)
    t_elapsed = time.perf_counter() - t_start
    stat_file.write(f'{i+1}\t{sum(map(len, solutions[-1].values()))}\t{round(t_elapsed, 3)}\n')
stat_file.close()


cube_vertices = np.array([
    np.array([0.5, 0.5, 0.5]),
    np.array([-0.5, 0.5, 0.5]),
    np.array([0.5, -0.5, 0.5]),
    np.array([-0.5, -0.5, 0.5]),
    np.array([0.5, 0.5, -0.5]),
    np.array([-0.5, 0.5, -0.5]),
    np.array([0.5, -0.5, -0.5]),
    np.array([-0.5, -0.5, -0.5])
])

triangles = np.array([
    [0, 1, 2],
    [1, 2, 3],
    [4, 5, 6],
    [5, 6, 7],
    [0, 1, 4],
    [1, 4, 5],
    [2, 3, 6],
    [3, 6, 7],
    [0, 2, 4],
    [2, 4, 6],
    [1, 3, 5],
    [3, 5, 7]
])


def get_plot_cube(cube, x_offset, y_offset):
    plot_cube = cube_vertices + np.full(cube.shape, cube)
    plot_cube += np.full(plot_cube.shape, np.array([x_offset, y_offset, 0]))
    return np.swapaxes(plot_cube, 0, 1)


fig = mlab.figure()
for i, solution in enumerate(solutions):
    solution_array = np.concatenate(list(solution.values()))
    if not os.path.exists(f'cuboid_{i+1}/'):
        os.makedirs(f'cuboid_{i+1}/')
        os.makedirs(f'cuboid_{i+1}/seperate_models/')
    # big_fig = mlab.figure()
    stream = open(f'cuboid_{i+1}/all_cuboids_size_{i+1}.txt', 'w')
    stream.write(f'n = {i+1}')
    stream.write('\n')
    stream.write(f'{len(solution_array)} unique cuboids found')
    stream.write('\n\n')
    for j, cuboid in enumerate(solution_array):
        stream.write(f'{j+1}:\n')
        for cube in cuboid:
            stream.write(f'{str(cube)}\n')
            mlab.triangular_mesh(*get_plot_cube(cube, 0, 0), triangles, figure=fig)
            # mlab.triangular_mesh(*get_plot_cube(cube, 0, 7*j), triangles, figure=big_fig)
        stream.write('\n\n')
        mlab.savefig(f'cuboid_{i+1}/seperate_models/cuboid_{i+1}_{j+1}.obj', figure=fig)
        mlab.clf(figure=fig)
    stream.close()
    del stream
    # mlab.savefig(f'cuboid_{i+1}/all_cuboids_size_{i+1}.obj', figure=big_fig)
    # mlab.close(big_fig)
mlab.close(fig)