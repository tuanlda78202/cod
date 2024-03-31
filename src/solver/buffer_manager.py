def _save_rehearsal(rehearsal, dir, task, memory):
    all_dir = os.path.join(dir, "Buffer_T_" + str(task) + "_" + str(memory))
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f"Directroy created")

    with open(all_dir, "wb") as f:
        pickle.dump(rehearsal, f)
        print(colored(f"Save task buffer", "light_red", "on_yellow"))


def load_rehearsal(dir, task=None, memory=None):
    if dir is None:
        return None

    if task == None and memory == None:
        all_dir = dir
    else:
        all_dir = os.path.join(dir, "Buffer_T_" + str(task) + "_" + str(memory))
    print(f"load replay file name : {all_dir}")
    if os.path.exists(all_dir):
        with open(all_dir, "rb") as f:
            temp = pickle.load(f)
            print(
                colored(
                    f"********** Loading {task} tasks' buffer ***********",
                    "blue",
                    "on_yellow",
                )
            )
            return temp
    else:
        print(
            colored(
                f"not exist file. plz check your replay file path or existence",
                "blue",
                "on_yellow",
            )
        )


def _load_replay_buffer(self):
    """
    you should check more then two task splits. because It is used in incremental tasks
    1. criteria : tasks >= 2
    2. args.Rehearsal : True
    3. args.
    """
    load_replay = []
    rehearsal_classes = {}
    args = self.args
    for idx in range(self.start_task):
        load_replay.extend(self.Divided_Classes[idx])

    load_task = 0 if args.start_task == 0 else args.start_task - 1

    # * Load for Replay
    if args.Rehearsal:
        rehearsal_classes = load_rehearsal(
            args.Rehearsal_file, load_task, args.limit_image
        )
        try:
            if len(list(rehearsal_classes.keys())) == 0:
                print(f"No rehearsal file. Initialization rehearsal dict")
                rehearsal_classes = {}
            else:
                print(f"replay keys length :{len(list(rehearsal_classes.keys()))}")
        except:
            print(f"Rehearsal File Error. Generate new empty rehearsal dict.")
            rehearsal_classes = {}

    return load_replay, rehearsal_classes
