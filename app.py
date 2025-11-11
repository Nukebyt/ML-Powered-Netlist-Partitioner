import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import torch
from itertools import combinations
import os
import sys
sys.path.append(os.path.dirname(__file__))

# Import our custom modules
try:
    from fm_partitioner import FMPartitioner
    from gnn_model import PartitionGNN, convert_to_graph_data, predict_initial_partition
    # Import the in-memory netlist generator so the UI does not write files to disk
    from netlist_generator2 import generate_random_netlist, generate_random_netlist_str
except Exception as e:
    import traceback
    st.error("Error importing modules. See details below:")
    st.error(str(e))
    st.text(traceback.format_exc())
    st.stop()


# --- Helper Functions ---

def parse_netlist_to_graph(file_content):
    """Parses the uploaded netlist text into a netlist and a NetworkX graph."""
    netlist = []
    all_cells = set()

    for line in file_content.splitlines():
        if line.strip() and not line.strip().startswith('#'):
            cells = line.strip().split()
            netlist.append(cells)
            all_cells.update(cells)

    G = nx.Graph()
    G.add_nodes_from(all_cells)

    for net in netlist:
        for u, v in combinations(net, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)

    return netlist, list(all_cells), G


def draw_partition(G, partitions, title):
    fig, ax = plt.subplots()
    color_map = ['skyblue' if node in partitions['A'] else 'lightgreen' for node in G.nodes()]
    pos = nx.spring_layout(G, k=0.8)

    nx.draw(G, pos, node_color=color_map, with_labels=True, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


@st.cache_resource
def load_model(model_path, num_features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PartitionGNN(num_node_features=num_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


# --- MAIN APP ---

st.set_page_config(layout="wide")
st.title("🧠 ML-Guided Circuit Partitioning (Fiduccia-Mattheyses)")


# 0. Model Selection
st.header("Choose a GNN Model")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

all_models = {
    "Model 2.0 OG (gnn_model2.pth)": os.path.join(BASE_DIR, "gnn_model2.pth"),
    "Model 1.0 (gnn_model1.pth)": os.path.join(BASE_DIR, "gnn_model1.pth")
}

existing_models = {k: v for k, v in all_models.items() if os.path.exists(v)}

if len(existing_models) == 0:
    st.error("❌ No .pth models found in folder.")
    st.stop()

selected_model_name = st.selectbox("Select trained model:", list(existing_models.keys()))
MODEL_PATH = existing_models[selected_model_name]


# --- 1. Netlist Input Section (Clean Layout) ---
st.header("1. Provide a Netlist")

col_left, col_right = st.columns(2)

with col_left:
    uploaded_file = st.file_uploader("📁 Upload Netlist (.txt)", type="txt")

    st.write("OR select a test netlist from folder:")

    # ✅ Use the correct folder name
    TEST_NETLIST_DIR = os.path.join(os.getcwd(), "sample netlists")

    if not os.path.exists(TEST_NETLIST_DIR):
        os.makedirs(TEST_NETLIST_DIR)

    txt_files = [f for f in os.listdir(TEST_NETLIST_DIR) if f.endswith(".txt")]

    if len(txt_files) > 0:
        selected_test_file = st.selectbox(
            "Available Test Netlists:",
            ["None"] + txt_files,
            key="test_netlist_select"
        )

        if selected_test_file != "None":
            try:
                path = os.path.join(TEST_NETLIST_DIR, selected_test_file)
                with open(path, "r", encoding="utf-8") as f:
                    st.session_state["file_content"] = f.read()
                st.success(f"✅ Loaded test netlist: {selected_test_file}")
            except Exception as e:
                st.error("Failed to load selected netlist:")
                st.error(str(e))
    else:
        st.info("No test netlists found in `/sample netlists` folder.")

with col_right:
    st.write("OR generate a random netlist:")
    num_cells = st.number_input("Cells", min_value=2, value=20)
    num_nets = st.number_input("Nets", min_value=1, value=30)

    if st.button("⚙ Generate Netlist"):
        try:
            # Use the in-memory generator to avoid creating files on disk
            netlist_text = generate_random_netlist_str(int(num_cells), int(num_nets))
            st.session_state["file_content"] = netlist_text
            st.success(f"✅ Generated netlist in-memory ({int(num_cells)} cells, {int(num_nets)} nets)")

        except Exception as e:
            st.error("Generation failed:")
            st.error(str(e))


# Uploaded file handler
if uploaded_file is not None:
    st.session_state["file_content"] = uploaded_file.getvalue().decode("utf-8")


# Stop if no netlist chosen
if "file_content" not in st.session_state:
    st.info("➡ Upload a netlist, select one, or generate one to begin.")
    st.stop()

file_content = st.session_state["file_content"]
netlist, all_cells, G = parse_netlist_to_graph(file_content)
st.write(f"Netlist detected: **{len(all_cells)} cells**, **{len(netlist)} nets**")


# --- Run FM ---
if st.button("🚀 Run Partitioning Analysis", type="primary"):

    with st.spinner("Running Standard FM (Random)..."):
        st.subheader("2. Standard FM (Random Start)")
        fm_random = FMPartitioner(netlist)
        partitions_random, cut_size_random = fm_random.partition(max_passes=10)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Final Cut Size (Random)", cut_size_random)
        with col2:
            draw_partition(G, partitions_random, "Final Partition (Random Start)")

    st.divider()

    with st.spinner("Running GNN-Guided FM..."):
        st.subheader("3. FM with GNN-Guided Start")
        model, device = load_model(MODEL_PATH, num_features=2)

        ml_initial_partition = predict_initial_partition(netlist, all_cells, model, device)

        fm_ml = FMPartitioner(netlist)
        partitions_ml, cut_size_ml = fm_ml.partition(
            max_passes=10,
            initial_partition=ml_initial_partition
        )

        col3, col4 = st.columns([1, 2])
        with col3:
            st.metric("Final Cut Size (ML-Guided)", cut_size_ml)
        with col4:
            draw_partition(G, partitions_ml, "Final Partition (GNN Start)")

    st.divider()

    st.subheader("📊 4. Comparison & Improvement")
    improvement = cut_size_random - cut_size_ml
    delta_percent = (improvement / cut_size_random) * 100 if cut_size_random > 0 else 0

    st.metric(
        label="Improvement in Cut Size",
        value=f"{improvement} cuts",
        delta=f"{delta_percent:.2f}%"
    )

    if improvement > 0:
        st.success("✅ GNN start produced a better partition!")
    elif improvement == 0:
        st.warning("⚠ Same result as random.")
    else:
        st.error("❌ GNN start was worse.")
