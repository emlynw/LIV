import csv, os, argparse, re

def count_frames(folder):
    # assumes filenames are 0.png, 1.png, …
    return len([f for f in os.listdir(folder) if re.match(r'\d+\.(png|jpg)', f)])

def write_manifest(root, subdir, success_flag, writer):
    demos = sorted(os.listdir(os.path.join(root, subdir)))
    for d in demos:
        folder = os.path.abspath(os.path.join(root, subdir, d))
        if not os.path.isdir(folder):
            continue
        n = count_frames(folder)
        if success_flag == 1:
            writer.writerow({
                'directory': folder,
                'num_frames': n,
                'text': "Pick red strawberry",  # crude task name
                'success': success_flag
            })
        else:
            writer.writerow({
                'directory': folder,
                'num_frames': n,
                'text': "Fail pick red strawberry",  # crude task name
                'success': success_flag
            })

def main(path):
    with open(os.path.join(path, 'manifest_success.csv'), 'w', newline='') as f_s, \
         open(os.path.join(path, 'manifest_failure.csv'), 'w', newline='') as f_f:
        fieldnames = ['directory', 'num_frames', 'text', 'success']
        ws, wf = csv.DictWriter(f_s, fieldnames), csv.DictWriter(f_f, fieldnames)
        ws.writeheader(); wf.writeheader()
        write_manifest(path, 'success', 1, ws)
        write_manifest(path, 'failure', 0, wf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', help='e.g. ./my_dataset')
    main(parser.parse_args().dataset_root)
