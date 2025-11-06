#!/usr/bin/env python3

import subprocess
import sys

def main():
    cmd = [
        "ign", "topic",
        "-t", "/model/rotating_table/joint/table_rotation/cmd_force",
        "-m", "ignition.msgs.Double",
        "-p", "data: 10.0"
    ]

    try:
        print("Sending force command to rotating table...")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout:
            print("Output:", result.stdout)

        print("Force command sent successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while sending the force command - {e}.")
        print("Error:", e.stderr)
        sys.exit(1)

    except FileNotFoundError:
        print("Error: Ignition command not found.")
        sys.exit(1)

    # if result.returncode != 0:
    #     print("Error:", result.stderr)
    #     sys.exit(1)

    print("Success:", result.stdout)

if __name__ == "__main__":
    main()