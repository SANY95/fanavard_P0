""" python 3 code format
Notes:
    numba is a JIT and multiprocessing is a costly operation to start so in smaller search spaces
    worker should be even slower than naive brute force but in larger search space it can be faster in scale of THR_core
"""

from numba import njit, prange
from numba.typed import List


@njit(parallel=True)
def worker(arr, m, k):
    """ worker method for automatic parallelization by JIT
    Args:
        arr (List): a list of a_i
        m (int): number of boxes
        k (int): size of every box
    Returns:
        (list): a list of boxed items in every start position
    Notes:
        prange is splinted version of range so mutex on response is not required
    """

    # memory allocation
    response = [0] * len(arr)

    # terminate search variables
    failed_count, failed_j = 0, -1

    for j in prange(0, len(arr)):

        # backward indexing
        j = len(arr) - j - 1

        # terminating search loop if there is no more valid answers
        if j < failed_j:
            continue

        c_sum = 0.0  #: current iteration sum of items
        c_m, c_i = 0, 0  #: number of used boxes and number of in box items

        # region sliding window
        i = j
        last = False  #: is set true if the last item is packed too
        while i < len(arr):

            if (c_sum + arr[i]) < k:
                # item can goes inside the box and still there is some room left
                c_sum += arr[i]
                c_i += 1
                if i == len(arr) - 1:
                    last = True
            else:
                # box is full or over loaded
                if (c_sum + arr[i]) > k:
                    # make sure box is not overloading
                    i -= 1
                else:
                    #  it's just a lucky full box
                    c_i += 1
                    if i == len(arr) - 1:
                        last = True

                # reset window parameters
                c_sum = 0
                c_m += 1

            # there is no more empty box
            if c_m == m:
                break
            i += 1

        # endregion

        # update the answer r and termination values
        if last:
            response[j] = c_i
        else:
            failed_count += 1
            if failed_count >= k:
                failed_j = j

    return response


def main():
    """ this is the main method for getting input from user and print the answer """

    inp = input().split(' ')
    n, m, k = int(inp[0]), int(inp[1]), int(inp[2])
    inp = input().split(' ')
    arr = [float(i) for i in inp]

    if n == 0 or m == 0 or k == 0:
        print('0')

    response = worker(List(arr), m, k)
    print(max(response))


def test():
    """ test method for pytest only so only using in CI test not for code call out """

    import time
    from random import randrange

    # some large inputs
    n = 2000
    m = 100
    k = 10
    arr = [randrange(1, k) for _ in range(n)]

    # timing test
    t0 = time.time()
    for _ in range(10):
        worker(List(arr), m, k)
    print(f'avg time={(time.time() - t0) / 10.0:.6f}')


if __name__ == '__main__':
    main()
