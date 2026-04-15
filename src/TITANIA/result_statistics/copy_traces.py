import os
def main(date):
    if not os.path.isdir("traces"):
        os.makedirs("traces")
    if not os.path.isdir("traces/example"):
        os.makedirs("traces/example")
    if not os.path.isdir("traces/example/dataset=Adult"):
        os.makedirs("traces/example/dataset=Adult")
    print(f"./outputs/FL_non_iid_settings/Adult/{date}")
    for (dirpath, dirnames, filenames) in os.walk(f"./outputs/FL_non_iid_settings/Adult/{date}"):
        new_path = dirpath.replace(f"./outputs/FL_non_iid_settings/Adult/{date}","traces/example/dataset=Adult")
        if not os.path.isdir(new_path):
            os.makedirs(new_path)
        for file in filenames:
            try:
                os.remove(new_path+"/"+file)
            except:
                pass
            with open(dirpath+"/"+file, "r") as f:
                with open(new_path+"/"+file, "w") as f2:
                    a="blank"
                    while a!="":
                        a=f.read()
                        f2.write(a)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--date', required=True, help='date of file name, for example 2026-04-14_11-44-41')
    args = parser.parse_args()
    main(args.date)
