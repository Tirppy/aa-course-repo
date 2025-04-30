import math
import random
import heapq

import tkinter as tk
from tkinter import ttk, messagebox

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class GraphUI:
    def __init__(self, root):
        root.title("Graph Algorithms Visualizer")

        # PanedWindow setup
        paned = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=1, padx=5, pady=5)

        left = ttk.Frame(paned, width=300)
        right = ttk.Frame(paned)
        paned.add(left); paned.add(right)
        paned.paneconfigure(left, pady=10); paned.paneconfigure(right, pady=10)

        # Control row: graph type + node count + start/end on one line
        ctrl_frame = ttk.Frame(left)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=(0,10))

        ttk.Label(ctrl_frame, text="Graph type:").grid(row=0, column=0, sticky="w")
        self.type_cb = ttk.Combobox(ctrl_frame,
                                    values=["Tree","Sparse","Dense","Complete","Grid"],
                                    state="readonly", width=10)
        self.type_cb.current(0)
        self.type_cb.grid(row=0, column=1, sticky="w", padx=(5,20))

        ttk.Label(ctrl_frame, text="Nodes:").grid(row=0, column=2, sticky="w")
        self.node_var = tk.IntVar(value=10)
        self.node_sb = ttk.Spinbox(ctrl_frame, from_=1, to=100,
                                textvariable=self.node_var,
                                width=5, justify="center")
        self.node_sb.grid(row=0, column=3, sticky="w", padx=5)

        # Start and End node entries
        ttk.Label(ctrl_frame, text="Start:").grid(row=1, column=0, sticky="w", pady=(5,0))
        self.start_var = tk.IntVar(value=0)
        ttk.Entry(ctrl_frame, textvariable=self.start_var, width=5).grid(
            row=1, column=1, sticky="w", pady=(5,0))  # Entry widget :contentReference[oaicite:3]{index=3}

        ttk.Label(ctrl_frame, text="End:").grid(row=1, column=2, sticky="w", pady=(5,0))
        self.end_var = tk.IntVar(value=0)
        ttk.Entry(ctrl_frame, textvariable=self.end_var, width=5).grid(
            row=1, column=3, sticky="w", pady=(5,0))  # grid placement :contentReference[oaicite:4]{index=4}

        # Weighted edges checkbox
        self.weight_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="Weighted Edges",
                        variable=self.weight_var).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(5,0))
        
        self.directed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ctrl_frame,
            text="Directed",
            variable=self.directed_var
        ).grid(row=2, column=4, sticky="w", padx=(10,0))  # new column for “Directed”

        # Adjacency list input box
        self.input_box = tk.Text(left, height=8)
        self.input_box.pack(fill=tk.X, padx=5, pady=5)

        # Generate / Push buttons
        btn_frame = ttk.Frame(left)
        btn_frame.pack(pady=5)
        for c in range(2): btn_frame.columnconfigure(c, weight=1)
        ttk.Button(btn_frame, text="Generate", command=self.on_generate)\
            .grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(btn_frame, text="Push", command=self.on_push)\
            .grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # -- Algorithm buttons (unchanged) --
        algs = ["DFS","BFS","Dijkstra","Floyd–Warshall","Prim","Kruskal"]
        alg_frame = ttk.LabelFrame(left, text="Run Algorithm")
        alg_frame.pack(fill=tk.X, padx=5, pady=10)
        for c in range(3):
            alg_frame.columnconfigure(c, weight=1)
        for idx, name in enumerate(algs):
            b = ttk.Button(alg_frame, text=name, command=lambda n=name: self.on_run(n))
            r, c = divmod(idx, 3)
            b.grid(row=r, column=c, padx=5, pady=5, sticky="ew")

        # -- Compare Labs (unchanged) --
        cmp_frame = ttk.LabelFrame(left, text="Compare Labs")
        cmp_frame.pack(fill=tk.X, padx=5, pady=10)
        for c in range(3):
            cmp_frame.columnconfigure(c, weight=1)
        ttk.Button(cmp_frame, text="Lab 1: DFS vs BFS",
                   command=lambda: self.on_compare(1)).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(cmp_frame, text="Lab 2: Dijkstra vs Floyd–Warshall",
                   command=lambda: self.on_compare(2)).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(cmp_frame, text="Lab 3: Prim vs Kruskal",
                   command=lambda: self.on_compare(3)).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        # -- Compare Settings (new) --
        settings = ttk.Frame(left)
        settings.pack(fill=tk.X, padx=5, pady=(0,10))
        ttk.Label(settings, text="Min Nodes:").grid(row=0, column=0)
        self.min_var = tk.IntVar(value=5)
        ttk.Spinbox(settings, from_=1, to=200, textvariable=self.min_var, width=5).grid(row=0, column=1)
        ttk.Label(settings, text="Max Nodes:").grid(row=0, column=2)
        self.max_var = tk.IntVar(value=50)
        ttk.Spinbox(settings, from_=1, to=200, textvariable=self.max_var, width=5).grid(row=0, column=3)
        ttk.Label(settings, text="Step:").grid(row=0, column=4)              
        self.step_var = tk.IntVar(value=5)                                
        ttk.Spinbox(settings, from_=1, to=100, textvariable=self.step_var, width=5).grid(row=0, column=5)
        ttk.Label(settings, text="Reps:").grid(row=0, column=6)
        self.rep_var = tk.IntVar(value=10)
        ttk.Spinbox(settings, from_=1, to=100, textvariable=self.rep_var, width=5).grid(row=0, column=7)

        # -- Right panel for drawing canvas --
        self.canvas_frame = right
        self.canvas = None  # will hold our FigureCanvasTkAgg

    def on_generate(self):

        n        = self.node_var.get()
        gtype    = self.type_cb.get()
        weighted = self.weight_var.get()
        directed = self.directed_var.get()

        # 1. Generate raw graph
        if gtype == "Sparse":
            # low‐density Erdős–Rényi
            p = 0.1
            if directed:
                # ensure strongly connected
                while True:
                    G = nx.gnp_random_graph(n=n, p=p, directed=True)
                    if nx.is_strongly_connected(G):
                        break
            else:
                while True:
                    G = nx.gnp_random_graph(n=n, p=p)
                    if nx.is_connected(G):
                        break

        elif gtype == "Dense":
            # high‐density Erdős–Rényi
            p = 0.9
            if directed:
                while True:
                    G = nx.gnp_random_graph(n=n, p=p, directed=True)
                    if nx.is_strongly_connected(G):
                        break
            else:
                while True:
                    G = nx.gnp_random_graph(n=n, p=p)
                    if nx.is_connected(G):
                        break

        elif gtype == "Complete":
            G = nx.complete_graph(n, create_using=nx.DiGraph() if directed else nx.Graph())

        elif gtype == "Grid":
            r = int(math.sqrt(n)) or 1
            c = math.ceil(n / r)
            H = nx.grid_2d_graph(r, c)
            if directed:
                H = nx.DiGraph(H)
            mapping = {(i, j): i*c + j for i, j in H.nodes()}
            G = nx.relabel_nodes(H, mapping)
            G.remove_nodes_from([u for u in G if u >= n])

        else:  # Tree
            T = nx.random_labeled_tree(n)
            if not nx.is_tree(T):
                messagebox.showwarning("Generation error", "Tree test failed")
            if directed:
                G = nx.DiGraph()
                G.add_nodes_from(T.nodes())
                root = self.start_var.get()
                for u, v in nx.bfs_edges(T, source=root):
                    G.add_edge(u, v)
            else:
                G = T

        # 2. Relabel nodes 0…n−1
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")  # :contentReference[oaicite:9]{index=9}

        # 3. Optionally assign random weights
        if weighted:
            w = {(u, v): random.randint(1, 10) for u, v in G.edges()}
            nx.set_edge_attributes(G, w, "weight")  # :contentReference[oaicite:10]{index=10}

        # 4. Show adjacency (with weights if any)
        self.input_box.delete("1.0", tk.END)
        for u in sorted(G.nodes()):
            nbrs = G.adj[u]
            if weighted:
                line = f"{u}: {sorted((v, nbrs[v]['weight']) for v in nbrs)}\n"
            else:
                line = f"{u}: {sorted(nbrs)}\n"
            self.input_box.insert(tk.END, line)

    def on_push(self):

        # 1. Parse adjacency + weights
        text = self.input_box.get("1.0", tk.END).strip().splitlines()
        adj, weights = {}, {}
        for line in text:
            if ":" not in line:
                continue
            u_str, rest = line.split(":", 1)
            u = int(u_str.strip())
            if "(" in rest:
                pairs = eval(rest, {}, {})
                adj[u] = [v for v, w in pairs]
                for v, w in pairs:
                    weights[(u, v)] = w
            else:
                adj[u] = eval(rest, {}, {})

        # 2. Build directed or undirected graph
        if self.directed_var.get():
            G = nx.DiGraph(adj)               # directed :contentReference[oaicite:13]{index=13}
        else:
            G = nx.Graph(adj)
        if weights:
            nx.set_edge_attributes(G, weights, "weight")  # :contentReference[oaicite:14]{index=14}

        # 3. Compute layout
        gtype = self.type_cb.get()
        if gtype == "Tree":
            def hierarchy_pos(G, root=0, width=1., vert_gap=0.2, vert_loc=0.):
                pos = {}
                def _hierarchy(n, left, right, vloc, parent=None):
                    pos[n] = ((left+right)/2, vloc)
                    children = [c for c in G.successors(n) if c != parent] if G.is_directed() else [c for c in G.neighbors(n) if c != parent]
                    if children:
                        span = (right-left)/len(children)
                        for i, c in enumerate(children):
                            _hierarchy(c, left+i*span, left+(i+1)*span, vloc-vert_gap, n)
                _hierarchy(self.start_var.get(), 0, width, vert_loc)
                return pos
            pos = hierarchy_pos(G)
        elif gtype == "Grid":
            n = self.node_var.get()
            r = int(math.sqrt(n)) or 1; c = math.ceil(n/r)
            pos = {u: (u % c, u // c) for u in G.nodes()}
        elif gtype == "Complete":
            pos = nx.circular_layout(G)       # :contentReference[oaicite:15]{index=15}
        else:
            pos = nx.spring_layout(G)

        # 4. Draw graph with updated weights
        fig = Figure(figsize=(4,4), dpi=100)
        ax  = fig.add_subplot(111)
        nx.draw_networkx(G, pos=pos, ax=ax, with_labels=True)
        edge_labels = nx.get_edge_attributes(G, "weight")
        if edge_labels:
            # move labels toward source end (0.1 of the edge length)
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                ax=ax,
                label_pos=0.1
            )

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # 5. Store for on_run
        self.current_G   = G
        self.current_pos = pos             # now correctly stored
        self.current_ax  = ax


    def on_run(self, algorithm_name):

        if algorithm_name == "Prim":
            self.run_prim_animate()
            return
        elif algorithm_name == "Kruskal":
            self.run_kruskal_animate()
            return
        elif algorithm_name == "Dijkstra":
            self.run_dijkstra_animate()
        elif algorithm_name == "Floyd–Warshall":
            self.run_floyd_animate()
        else:

            # 1. Ensure graph & layout exist
            if not hasattr(self, 'current_G') or not hasattr(self, 'current_pos'):
                self.on_push()
            G   = self.current_G
            pos = self.current_pos
            ax  = self.current_ax
            widget = self.canvas.get_tk_widget()

            start = self.start_var.get()
            end   = self.end_var.get()

            # Prepare the sequence of tree-edges and node-order
            edges = []
            nodes_order = []

            try:
                if algorithm_name == "BFS":
                    all_edges = list(nx.bfs_edges(G, source=start))
                    nodes_order = [start] 
                    for u, v in all_edges:
                        edges.append((u, v))
                        nodes_order.append(v)
                        if v == end:
                            break

                elif algorithm_name == "DFS":
                    nodes_order = [start]
                    for u, v in nx.dfs_edges(G, source=start):
                        edges.append((u, v))
                        nodes_order.append(v)
                        if v == end:
                            break

                elif algorithm_name == "Dijkstra":
                    # Compute shortest path (returns list of nodes) :contentReference[oaicite:2]{index=2}
                    path = nx.dijkstra_path(G, source=start, target=end, weight='weight')
                    nodes_order = path
                    # zip yields an iterator; no need for list() unless you need indexing :contentReference[oaicite:3]{index=3}
                    edges = zip(path, path[1:])

                elif algorithm_name == "Floyd–Warshall":
                    # Ensure graph is drawn & stored
                    if not hasattr(self, 'current_G') or not hasattr(self, 'current_pos'):
                        self.on_push()
                    self.run_floyd_animate()
                    return

                else:
                    return

            except nx.NetworkXNoPath:
                messagebox.showwarning("No Path", f"No path from {start} to {end}")
                return
            except nx.NodeNotFound as e:
                messagebox.showwarning("Node Error", str(e))
                return

            # 2. Draw base graph in gray
            ax.clear()
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray')  # base nodes
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray')       # base edges
            nx.draw_networkx_labels(G, pos, ax=ax)                         # labels
            # redraw weights if present
            edge_labels = nx.get_edge_attributes(G, 'weight')
            if edge_labels:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)  # show weights
            ax.set_axis_off()
            self.canvas.draw()

            # 3. Animate: highlight each node in red and its tree-edge in blue
            def step(i):
                if i >= len(nodes_order):
                    return
                u = nodes_order[i]
                # highlight node
                nx.draw_networkx_nodes(G, pos, nodelist=[u], node_color='red', ax=ax)
                # highlight its incoming edge
                if i > 0:
                    e = edges[i-1]
                    nx.draw_networkx_edges(G, pos, edgelist=[e],
                                        edge_color='blue', width=2, ax=ax)
                self.canvas.draw()
                widget.after(500, lambda: step(i+1))

            step(0)

    def animate_events(self, events):
        """Animate a list of events on self.current_ax with self.current_pos layout."""

        G      = self.current_G
        pos    = self.current_pos
        ax     = self.current_ax
        widget = self.canvas.get_tk_widget()
        start, end = self.start_var.get(), self.end_var.get()

        def step(i):
            ax.clear()
            # — draw base graph —
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray')
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray')
            nx.draw_networkx_labels(G, pos, ax=ax)
            # highlight start/end
            nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='green', ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=[end],   node_color='red',   ax=ax)
            # redraw weights at start of edges
            edge_labels = nx.get_edge_attributes(G, 'weight')
            if edge_labels:
                nx.draw_networkx_edge_labels(
                    G,
                    pos,
                    edge_labels=edge_labels,
                    ax=ax,
                    label_pos=0.1
                )

            if i < len(events):
                typ, *data = events[i]
                if typ == "consider":
                    u, v = data
                    nx.draw_networkx_edges(G, pos, edgelist=[(u,v)],
                                           edge_color='yellow', width=3, ax=ax)
                    nx.draw_networkx_nodes(G, pos, nodelist=[v],
                                           node_color='orange', ax=ax)
                elif typ == "update":
                    u, v = data
                    nx.draw_networkx_edges(G, pos, edgelist=[(u,v)],
                                           edge_color='blue', width=3, ax=ax)
                    nx.draw_networkx_nodes(G, pos, nodelist=[v],
                                           node_color='blue', ax=ax)
                elif typ == "consider_fw":
                    i_node, k, j = data
                    nx.draw_networkx_nodes(G, pos, nodelist=[k],
                                           node_color='orange', ax=ax)
                    nx.draw_networkx_edges(G, pos,
                          edgelist=[(i_node,k),(k,j)],
                          edge_color='yellow', width=3, ax=ax)
                elif typ == "update_fw":
                    i_node, j, k = data
                    nx.draw_networkx_nodes(G, pos, nodelist=[j],
                                           node_color='blue', ax=ax)
                    nx.draw_networkx_edges(G, pos, edgelist=[(i_node,j)],
                                           edge_color='blue', width=3, ax=ax)
                elif typ == "final":
                    u, v = data
                    nx.draw_networkx_edges(G, pos, edgelist=[(u,v)],
                                           edge_color='green', width=4, ax=ax)

                self.canvas.draw()
                widget.after(300, lambda: step(i+1))
            else:
                # once all events done, draw entire final path
                if hasattr(self, 'final_path_edges'):
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=self.final_path_edges,
                        edge_color='green',
                        width=4,
                        ax=ax
                    )
                    self.canvas.draw()
                # then show the length prompt only once
                messagebox.showinfo(
                    "Result",
                    f"Shortest-path length from {start} to {end} is {self.last_path_length}"
                )
                # do not clear—leave final path highlighted

        step(0)

    def run_dijkstra_animate(self):

        G     = self.current_G
        start = self.start_var.get()
        end   = self.end_var.get()

        # initialize
        dist = {u: float('inf') for u in G.nodes()}
        prev = {u: None for u in G.nodes()}
        dist[start] = 0
        pq = [(0, start)]
        visited = set()

        events = []
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == end:
                break
            for v, data in G[u].items():
                w = data.get('weight', 1)
                # considering edge
                events.append(("consider", u, v))
                if dist[v] > d + w:
                    dist[v] = d + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
                    # relaxed edge
                    events.append(("update", u, v))

        # reconstruct path
        path_edges = []
        node = end
        while prev[node] is not None:
            path_edges.append((prev[node], node))
            node = prev[node]
        path_edges.reverse()

        # store for final highlighting
        self.final_path_edges = path_edges
        self.last_path_length   = dist[end] if dist[end] < float('inf') else None

        # now animate
        self.animate_events(events)

    def run_floyd_animate(self):
        G = self.current_G
        nodes = list(G.nodes())

        # fetch start/end here to avoid NameError :contentReference[oaicite:3]{index=3}
        start, end = self.start_var.get(), self.end_var.get()

        # initialize
        dist = {u:{v: float('inf') for v in nodes} for u in nodes}
        next_hop = {u:{v: None for v in nodes} for u in nodes}
        for u in nodes:
            dist[u][u] = 0
            next_hop[u][u] = u
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 1)
            dist[u][v] = w
            next_hop[u][v] = v
            if not G.is_directed():
                dist[v][u] = w
                next_hop[v][u] = u

        events = []
        # triple loop
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    events.append(("consider_fw", i, k, j))
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_hop[i][j] = next_hop[i][k]
                        events.append(("update_fw", i, j, k))

        # reconstruct single start→end path
        path_edges = []
        u = start
        while u != end and next_hop[u][end] is not None:
            v = next_hop[u][end]
            path_edges.append((u, v))
            u = v

        # store for final highlighting
        self.final_path_edges = path_edges
        self.last_path_length  = dist[start][end] if dist[start][end] < float('inf') else None

        # animate
        self.animate_events(events)

    def run_prim_animate(self):

        G     = self.current_G
        start = self.start_var.get()
        end   = self.end_var.get()

        # --- 1. build MST with Prim, record events ---
        in_mst = {start}
        best   = {v:(float('inf'),None) for v in G.nodes()}
        for v, data in G[start].items():
            best[v] = (data.get('weight',1), start)
        pq = [(w,v) for v,(w,_) in best.items() if v!=start]
        heapq.heapify(pq)

        events = []
        mst_edges = []

        while pq:
            w,u = heapq.heappop(pq)
            if best[u][0]!=w: 
                continue
            prev = best[u][1]
            # add u to MST via prev→u
            in_mst.add(u)
            mst_edges.append((prev,u))
            events.append(("final", prev, u))
            # relax out-edges of u
            for nbr, data in G[u].items():
                if nbr in in_mst: continue
                wt = data.get('weight',1)
                events.append(("consider", u, nbr))
                if wt < best[nbr][0]:
                    best[nbr] = (wt,u)
                    events.append(("update", u, nbr))
                    heapq.heappush(pq,(wt,nbr))

        # --- 2. extract the path within the MST from start to end ---
        T = nx.DiGraph() if G.is_directed() else nx.Graph()
        T.add_edges_from(mst_edges)
        path = nx.shortest_path(T, source=start, target=end)
        route = list(zip(path, path[1:]))

        # store for final highlighting & prompt
        self.final_path_edges = route
        self.last_path_length = sum(G[u][v].get('weight',1) for u,v in route)

        # --- 3. animate ---
        self.animate_events(events)

    def run_kruskal_animate(self):

        G     = self.current_G
        start = self.start_var.get()
        end   = self.end_var.get()

        # ── 1. Build MST via Kruskal (union–find) ──
        parent = {u: u for u in G.nodes()}
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        def union(u, v):
            parent[find(v)] = find(u)

        # sort edges by weight
        all_edges = list(G.edges(data=True))
        all_edges.sort(key=lambda x: x[2].get('weight', 1))

        events = []
        mst_edges = []

        for u, v, data in all_edges:
            events.append(("consider", u, v))
            if find(u) != find(v):
                union(u, v)
                events.append(("update", u, v))
                mst_edges.append((u, v))

        # highlight MST edges at the end
        for u, v in mst_edges:
            events.append(("final", u, v))

        # ── 2. Try to extract the start→end route within the MST ──
        # build T as an undirected graph for connectivity
        T = nx.Graph()
        T.add_edges_from(mst_edges)

        try:
            path = nx.shortest_path(T, source=start, target=end)
            route = list(zip(path, path[1:]))
            # store for final highlighting & length prompt
            self.final_path_edges = route
            # Safely look up weight in either direction:
            total = 0
            for u, v in route:
                data = G.get_edge_data(u, v)
                if data is None:
                    data = G.get_edge_data(v, u)
                total += data.get('weight', 1)
            self.last_path_length = total
        except nx.NetworkXNoPath:
            # no path in the MST between start and end
            self.final_path_edges = []
            self.last_path_length = None
            messagebox.showwarning(
                "No route in MST",
                f"No path between {start} and {end} in the minimum spanning forest."
            )

        # ── 3. Animate all the “consider/update/final” events ──
        self.animate_events(events)

    def on_compare(self, lab_number):
        import time
        import networkx as nx

        min_n = self.min_var.get()
        max_n = self.max_var.get()
        step  = self.step_var.get()
        reps  = self.rep_var.get()
        x_vals = list(range(min_n, max_n + 1, step))

        # prepare plot
        fig = Figure(figsize=(6,4), dpi=100)
        ax  = fig.add_subplot(111)

        if lab_number == 1:
            # DFS vs BFS on random (dense) undirected graphs
            dfs_times = []
            bfs_times = []
            for n in x_vals:
                t_dfs = 0.0
                t_bfs = 0.0
                for _ in range(reps):
                    # generate a connected dense graph (p=0.9)
                    while True:
                        G = nx.gnp_random_graph(n=n, p=0.9)
                        if nx.is_connected(G):
                            break
                    # time DFS
                    t0 = time.perf_counter()
                    _ = list(nx.dfs_edges(G, source=0))
                    t_dfs += time.perf_counter() - t0
                    # time BFS
                    t0 = time.perf_counter()
                    _ = list(nx.bfs_edges(G, source=0))
                    t_bfs += time.perf_counter() - t0
                dfs_times.append(t_dfs/reps)
                bfs_times.append(t_bfs/reps)
            ax.plot(x_vals, dfs_times, marker='o', label='DFS')
            ax.plot(x_vals, bfs_times, marker='o', label='BFS')
            ax.set_title('Lab 1: DFS vs BFS')

        elif lab_number == 2:
            # Dijkstra vs Floyd–Warshall on random directed graphs
            dij_times = []
            fw_times  = []
            for n in x_vals:
                t_dij = 0.0
                t_fw  = 0.0
                for _ in range(reps):
                    # generate strongly connected directed G(n,0.3)
                    while True:
                        G = nx.gnp_random_graph(n=n, p=0.3, directed=True)
                        if nx.is_strongly_connected(G):
                            break
                    # assign random weights 1–10
                    w = {e: random.randint(1,10) for e in G.edges()}
                    nx.set_edge_attributes(G, w, 'weight')
                    # Dijkstra from 0 to n-1
                    t0 = time.perf_counter()
                    _ = nx.dijkstra_path(G, source=0, target=n-1, weight='weight')
                    t_dij += time.perf_counter() - t0
                    # Floyd–Warshall all-pairs
                    t0 = time.perf_counter()
                    _ = nx.floyd_warshall(G, weight='weight')
                    t_fw += time.perf_counter() - t0
                dij_times.append(t_dij/reps)
                fw_times.append(t_fw/reps)
            ax.plot(x_vals, dij_times, marker='o', label='Dijkstra')
            ax.plot(x_vals, fw_times,  marker='o', label='Floyd–Warshall')
            ax.set_title('Lab 2: Dijkstra vs Floyd–Warshall')

        elif lab_number == 3:
            # Prim vs Kruskal on random undirected weighted graphs
            prim_times   = []
            kruskal_times= []
            for n in x_vals:
                t_prim = 0.0
                t_krus  = 0.0
                for _ in range(reps):
                    # generate connected undirected G(n,0.5)
                    while True:
                        G = nx.gnp_random_graph(n=n, p=0.5)
                        if nx.is_connected(G):
                            break
                    # random weights
                    w = {e: random.randint(1,10) for e in G.edges()}
                    nx.set_edge_attributes(G, w, 'weight')
                    # time Prim
                    t0 = time.perf_counter()
                    _ = nx.minimum_spanning_tree(G, algorithm='prim', weight='weight')
                    t_prim += time.perf_counter() - t0
                    # time Kruskal
                    t0 = time.perf_counter()
                    _ = nx.minimum_spanning_tree(G, algorithm='kruskal', weight='weight')
                    t_krus += time.perf_counter() - t0
                prim_times.append(t_prim/reps)
                kruskal_times.append(t_krus/reps)
            ax.plot(x_vals, prim_times,    marker='o', label='Prim')
            ax.plot(x_vals, kruskal_times, marker='o', label='Kruskal')
            ax.set_title('Lab 3: Prim vs Kruskal')

        else:
            messagebox.showinfo("Compare", f"Lab {lab_number} not implemented.")
            return

        ax.set_xlabel('Number of nodes')
        ax.set_ylabel('Average time (s)')
        ax.legend()
        ax.grid(True)

        # embed in tkinter
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphUI(root)
    root.mainloop()