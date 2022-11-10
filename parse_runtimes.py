import os
import pandas as pd

"""Edited from:https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tree/roofline-hackathon-2020/"""

if __name__ == "__main__":
    datadir = ''
    files = ['profile_' + str(i) + '.out' for i in range(1, 11)]
    files = [os.path.join(datadir, file) for file in files]
    for file in files:
        tag, ext = os.path.splitext(os.path.basename(file))
        with open(file, 'r') as f:
            cnt = 0
            while True:
                ln = f.readline()
                if not ln:
                    break
                cnt += 1
                if 'Host Name' in ln:
                    break
            df = pd.read_csv(file, skiprows=cnt - 1)
            df = df[df['Kernel Name'] != 'centerOfMass(t_particles, float *, float *, float *, float *, int *, int)']

            dfmetric = dict(zip(df['Metric Name'], df['Metric Value'].str.replace(',', '').astype(float)))
            dfmetric['Count'] = 1

            dfmetric['Time'] = dfmetric['sm__cycles_elapsed.avg'] \
                               / (dfmetric['sm__cycles_elapsed.avg.per_second'] / dfmetric['Count'])

            dfmetric['CC FLOPs'] = 2 * dfmetric['sm__sass_thread_inst_executed_op_dfma_pred_on.sum'] \
                                   + dfmetric['sm__sass_thread_inst_executed_op_dmul_pred_on.sum'] \
                                   + dfmetric['sm__sass_thread_inst_executed_op_dadd_pred_on.sum'] \
                                   + 2 * dfmetric['sm__sass_thread_inst_executed_op_ffma_pred_on.sum'] \
                                   + dfmetric['sm__sass_thread_inst_executed_op_fmul_pred_on.sum'] \
                                   + dfmetric['sm__sass_thread_inst_executed_op_fadd_pred_on.sum'] \
                                   + 2 * dfmetric['sm__sass_thread_inst_executed_op_hfma_pred_on.sum'] \
                                   + dfmetric['sm__sass_thread_inst_executed_op_hmul_pred_on.sum'] \
                                   + dfmetric['sm__sass_thread_inst_executed_op_hadd_pred_on.sum']

            dfmetric['TC FLOPs'] = 512 * dfmetric['sm__inst_executed_pipe_tensor.sum']
            dfmetric['all FLOPs'] = dfmetric['CC FLOPs'] + dfmetric['TC FLOPs']

            dfmetric['MB/s'] = dfmetric['dram__bytes.sum.per_second'] / 1024

            dfmetric['MFLOP/s'] = dfmetric['all FLOPs'] / dfmetric['Time'] / 1024 / 1024

            print(f"{tag} stats")
            for key in ['Time', 'MB/s', 'MFLOP/s']:
                print(key, f"{dfmetric[key]:.6f}")
            print("")
