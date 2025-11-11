import os
import random


def _build_random_nets(num_cells, num_nets):
    """Helper: returns a sorted list of net strings (no file I/O)."""
    cells = [f"c{i}" for i in range(1, num_cells + 1)]
    nets = set()

    size_distribution = [2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10]

    while len(nets) < num_nets:
        size = random.choice(size_distribution)
        pick = random.sample(cells, min(size, len(cells)))
        formatted = " ".join(sorted(pick, key=lambda x: int(x[1:])))
        nets.add(formatted)

    # return a deterministic-ish ordering for readability
    return sorted(nets)


def generate_random_netlist_str(num_cells, num_nets):
    """Return a netlist as a string (no file I/O)."""
    nets = _build_random_nets(num_cells, num_nets)
    lines = [f"# Random Netlist: {num_cells} cells, {num_nets} nets"]
    lines += nets
    return "\n".join(lines) + "\n"


def generate_random_netlist(num_cells, num_nets, filename):
    """Backward-compatible: generate a netlist and save to filename (same behavior).

    Internally uses _build_random_nets so the core logic is reusable for in-memory
    generation in UIs.
    """
    nets = _build_random_nets(num_cells, num_nets)

    # make sure .txt exists
    if not filename.endswith(".txt"):
        filename += ".txt"

    # full path where file will be written
    output_path = os.path.join(os.getcwd(), filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Random Netlist: {num_cells} cells, {num_nets} nets\n")
        for net in nets:
            f.write(net + "\n")
        f.flush()

    print(f"✅ Random netlist saved to: {output_path}")


if __name__ == "__main__":
    num_cells = int(input("Enter number of cells: "))
    num_nets = int(input("Enter number of nets: "))

    filename = f"{num_cells}cells_{num_nets}nets_random.txt"
    generate_random_netlist(num_cells, num_nets, filename)
