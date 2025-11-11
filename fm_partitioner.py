import collections
import random

class FMPartitioner:
    """
    Implements the Fiduccia-Mattheyses (FM) algorithm for hypergraph partitioning.
    Modified to accept a pre-defined initial partition.
    """
    def __init__(self, netlist):
        self.cells = collections.defaultdict(lambda: {'nets': set(), 'partition': None, 'gain': 0, 'locked': False})
        self.nets = collections.defaultdict(lambda: {'cells': set()})
        self._parse_netlist(netlist)
        # Handle case where netlist might be empty or cells have no nets
        self.p_max = max((len(cell['nets']) for cell in self.cells.values()), default=0) if self.cells else 0
        self.gain_buckets = {'A': collections.defaultdict(set), 'B': collections.defaultdict(set)}
        self.max_gain = {'A': -self.p_max, 'B': -self.p_max}

    def _parse_netlist(self, netlist):
        """Parses a list-based netlist into the internal cell/net dictionary format."""
        for i, net_cells in enumerate(netlist):
            net_id = f"n{i}"
            for cell_id in net_cells:
                self.cells[cell_id]['nets'].add(net_id)
                self.nets[net_id]['cells'].add(cell_id)

    def _calculate_cut_size(self):
        """Calculates the number of nets that are cut by the current partition."""
        cut_size = 0
        for net_id in self.nets:
            partitions_spanned = {self.cells[cell_id]['partition'] for cell_id in self.nets[net_id]['cells']}
            if len(partitions_spanned) > 1:
                cut_size += 1
        return cut_size

    def _get_net_distribution(self, net_id):
        """Calculates the number of cells from a net in each partition."""
        dist = {'A': 0, 'B': 0}
        for cell_id in self.nets[net_id]['cells']:
            partition = self.cells[cell_id]['partition']
            if partition:
                dist[partition] += 1
        return dist

    def _compute_cell_gain(self, cell_id):
        """Computes the gain for a single cell."""
        gain = 0
        from_partition = self.cells[cell_id]['partition']
        to_partition = 'B' if from_partition == 'A' else 'A'
        for net_id in self.cells[cell_id]['nets']:
            dist = self._get_net_distribution(net_id)
            if dist[from_partition] == 1: gain += 1
            if dist[to_partition] == 0: gain -= 1
        return gain

    def _update_gains_after_move(self, moved_cell_id):
        """Updates the gains of all affected neighbor cells after a move."""
        from_partition = self.cells[moved_cell_id]['partition']
        to_partition = 'B' if from_partition == 'A' else 'A'
        # "Move" the cell
        self.cells[moved_cell_id]['partition'] = to_partition
        
        affected_nets = self.cells[moved_cell_id]['nets']
        cells_to_update = set()
        for net_id in affected_nets:
            for cell_id in self.nets[net_id]['cells']:
                cells_to_update.add(cell_id)
        
        for cell_id in cells_to_update:
            if not self.cells[cell_id]['locked']:
                old_gain = self.cells[cell_id]['gain']
                part = self.cells[cell_id]['partition']
                self.gain_buckets[part][old_gain].discard(cell_id)
                new_gain = self._compute_cell_gain(cell_id)
                self.cells[cell_id]['gain'] = new_gain
                self.gain_buckets[part][new_gain].add(cell_id)
                if new_gain > self.max_gain[part]:
                    self.max_gain[part] = new_gain
                elif old_gain == self.max_gain[part] and not self.gain_buckets[part][old_gain]:
                    self._update_max_gain(part)

    def _build_gain_buckets(self):
        """Initializes gain buckets for a new pass."""
        self.gain_buckets = {'A': collections.defaultdict(set), 'B': collections.defaultdict(set)}
        self.max_gain = {'A': -self.p_max, 'B': -self.p_max}
        for cell_id, cell_data in self.cells.items():
            cell_data['gain'] = self._compute_cell_gain(cell_id)
            partition = cell_data['partition']
            self.gain_buckets[partition][cell_data['gain']].add(cell_id)
            if cell_data['gain'] > self.max_gain[partition]:
                self.max_gain[partition] = cell_data['gain']
    
    def _update_max_gain(self, partition):
        """Finds the new highest-gain bucket that is not empty."""
        for g in range(self.max_gain[partition], -self.p_max - 1, -1):
            if self.gain_buckets[partition][g]:
                self.max_gain[partition] = g
                return
        self.max_gain[partition] = -self.p_max

    def _select_base_cell(self):
        """Selects the best unlocked cell to move based on gain and balance."""
        size_A = sum(1 for c in self.cells.values() if c['partition'] == 'A' and not c['locked'])
        size_B = sum(1 for c in self.cells.values() if c['partition'] == 'B' and not c['locked'])
        
        if size_A > size_B: partition_to_pick = 'A'
        elif size_B > size_A: partition_to_pick = 'B'
        else: partition_to_pick = 'A' if self.max_gain['A'] >= self.max_gain['B'] else 'B'
        
        max_g = self.max_gain[partition_to_pick]
        if self.gain_buckets[partition_to_pick][max_g]:
            return self.gain_buckets[partition_to_pick][max_g].pop()
        
        other_part = 'B' if partition_to_pick == 'A' else 'A'
        max_g_other = self.max_gain[other_part]
        if self.gain_buckets[other_part][max_g_other]:
            return self.gain_buckets[other_part][max_g_other].pop()
        
        return None

    def _run_pass(self):
        """Executes a single pass of the FM algorithm."""
        for cell in self.cells.values(): cell['locked'] = False
        self._build_gain_buckets()
        move_history, cut_history = [], [self._calculate_cut_size()]

        for _ in range(len(self.cells)):
            base_cell_id = self._select_base_cell()
            if base_cell_id is None: break
            self.cells[base_cell_id]['locked'] = True
            move_history.append((base_cell_id, self.cells[base_cell_id]['partition']))
            self._update_gains_after_move(base_cell_id)
            cut_history.append(self._calculate_cut_size())

        min_cut = min(cut_history)
        best_move_index = cut_history.index(min_cut)
        
        for i in range(len(move_history) - 1, best_move_index - 1, -1):
            cell_id, from_part = move_history[i]
            self.cells[cell_id]['partition'] = from_part
        return cut_history[0], min_cut

    def partition(self, max_passes=10, initial_partition=None):
        """
        Runs the full partitioning process.
        Accepts an optional 'initial_partition' dictionary.
        """
        if initial_partition:
            # Use the provided ML-guided partition
            for cell_id in self.cells:
                self.cells[cell_id]['partition'] = 'A' if cell_id in initial_partition['A'] else 'B'
        else:
            # Create a random partition
            cell_ids = list(self.cells.keys())
            random.shuffle(cell_ids)
            midpoint = len(cell_ids) // 2
            for i, cell_id in enumerate(cell_ids):
                self.cells[cell_id]['partition'] = 'A' if i < midpoint else 'B'

        # Run passes until no improvement
        for i in range(max_passes):
            initial_cut, min_cut = self._run_pass()
            if min_cut >= initial_cut:
                break
        
        final_partitions = {'A': [], 'B': []}
        for cell_id, data in self.cells.items():
            final_partitions[data['partition']].append(cell_id)
        return final_partitions, self._calculate_cut_size()