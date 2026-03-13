import os
from tensorboard import program

FILEPATH = os.path.abspath(__file__)

def main():
    """Launch tensorboard with the latest run in the logs folder."""
    logs_folder = os.path.join(FILEPATH, "..", "..", "..", "logs", "train", "runs")  # the path of your log folder.
    logs_folder = os.path.abspath(logs_folder)
    latest_run = max([os.path.join(logs_folder, f) for f in os.listdir(logs_folder) if os.path.isdir(os.path.join(logs_folder, f))], key=os.path.getmtime,)
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", os.path.join(latest_run, "tensorboard")])
    # url = tb.launch()
    # print(f"Tensorflow listening on {url}")
    tb.main()

if __name__ == "__main__":
    main()