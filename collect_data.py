import sys
import os
import argparse
import serial
import time
import numpy as np
from toolkit.utils import gen_acc_gyro_rmat
from halo import Halo


def read(s, num=1):
    res = b''
    while len(res) < num:
        res += s.read(num - len(res))
    return res


def drop_all_data(s):
    time.sleep(0.1)
    while(True):  # drop all
        _tmp = s.read(1)
        if(len(_tmp) == 0):
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--capture_sec", type=int, default=60 * 5)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--bias", type=str, required=False)
    parser.add_argument("--total_adjust_rmat", type=str, required=False)
    args = parser.parse_args()

    imu_n = 30  # 使用センサ数

    if args.bias is not None:
        bias = np.loadtxt(args.bias)
    else:
        bias = np.hstack((np.zeros((imu_n, 6)), np.ones((imu_n, 3))))

    if args.total_adjust_rmat is not None:
        total_adjust_rmat = np.load(args.total_adjust_rmat)
    else:
        total_adjust_rmat = np.array([np.eye(3) for i in range(imu_n)])

    total_rmat = gen_acc_gyro_rmat(total_adjust_rmat, bias=bias, debug=False)

    if args.capture_sec <= 0:
        # It is enough to not time-limited capture.
        args.capture_sec = 60 * 60 * 24 * 365

    with serial.Serial(args.port, 230400, timeout=0.2) as s:
        time.sleep(0.2)

        s.write(bytes('e', 'utf-8'))
        s.flush()
        drop_all_data(s)

        s.write(bytes('s', 'utf-8'))
        s.flush()
        print('Reading sensors')

        try:

            start_time = time.time()
            data = {
                'imu': [],
                'timestamp': [],
                'etime': [],
            }

            while time.time() - start_time < args.capture_sec:

                count_zero = 0
                while True:

                    n = int.from_bytes(read(s, 1), "little", signed=False)

                    if n == 0:

                        count_zero += 1
                    else:

                        count_zero = 0

                    if count_zero >= 2:

                        break
                inertia = [[int.from_bytes(read(s, 2), "big", signed=True) for _ in range(
                    6)] for sens_ix in range(imu_n)]  # 2*6 12 * 7 = 84
                timestamp = int.from_bytes(
                    read(s, 4), "little", signed=False)  # 4 : total 100
                etime = int.from_bytes(
                    read(s, 2), "little", signed=False)  # 2 total 102
                checksum = int.from_bytes(read(s, 1), "little", signed=False)

                txt_inertia = f'timestamp={str(timestamp): <16}\n'
                txt_inertia += f'{"":<8}'
                for metric in ['acc', 'gyr']:
                    for axis in list('xyz'):
                        l = f'{metric}.{axis}'
                        txt_inertia += f'{l:>8}'
                txt_inertia += '\n'
                for in_i, item in enumerate(inertia):
                    label = f'imu[{in_i}]:'
                    txt_inertia += f'{label:<8}'
                    for ii in item:
                        txt_inertia += f'{str(ii):>8}'
                    txt_inertia += '\n'
                lcount = txt_inertia.count('\n') + 1
                print(txt_inertia + f'\033[{lcount}A')

                # remove bias
                inertia = np.array(inertia, dtype=float) - bias[:, 0:6]

                # adjust axes of IMUs
                for dev_ix in range(inertia.shape[0]):
                    inertia[dev_ix, :] = np.dot(
                        inertia[dev_ix, :], total_rmat[dev_ix, :, :]
                    )
                inertia = inertia.reshape((1,) + inertia.shape)

                data['imu'].append(inertia)
                data['timestamp'].append(timestamp)
                data['etime'].append(etime)

        except KeyboardInterrupt:
            print('done!')
        finally:
            s.write(bytes('e', 'utf-8'))
            s.flush()
            drop_all_data(s)
            make_path = "/Users/matsuokoki/Desktop/output_rawdata"  # 　出力先
            os.chdir(make_path)
            for dkey, dval in data.items():
                res = np.stack(dval).squeeze()
                path = os.path.join(args.out, f'{dkey}.npy')
                os.makedirs(os.path.join(args.out), exist_ok=True)
                np.save(path, res)
                print(f'{dkey: <10} {str(res.shape): <10} -> {path}')
            print(((float(data["timestamp"][-1]) -
                  float(data["timestamp"][0])) / 1000000))


if __name__ == "__main__":
    main()
