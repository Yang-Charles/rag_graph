"""Simple knowledge graph service with optional Neo4j backing.

By default this service uses an in-memory NetworkX graph for quick local
testing. If a Neo4j URI/credentials are provided (via `connect_neo4j` or the
constructor), KG operations will be executed against Neo4j instead.

Implemented features:
- `kg_insert_nodes_edges(nodes, edges)` to insert/merge nodes and relationships into Neo4j.
- `search_entities(query, topk)` to search node `text` and `label` properties in Neo4j.
- `expand_neighbors(node_id, hops)` to get neighbors up to `hops` deep (Neo4j or NetworkX).
"""

import networkx as nx
from typing import List, Tuple, Optional, Dict, Any

try:
    from neo4j import GraphDatabase, basic_auth
except Exception:
    GraphDatabase = None


class KGService:
    def __init__(self, neo4j_uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.G = nx.Graph()
        self._build_sample()

        # Neo4j driver (optional)
        self.driver = None
        if neo4j_uri and GraphDatabase is not None:
            self.connect_neo4j(neo4j_uri, user, password)

    def _build_sample(self):
        # Small example graph for local fallback
        self.G.add_node(1, label="Document 1", text="关于产品A的说明")
        self.G.add_node(2, label="Document 2", text="关于产品B的说明")
        self.G.add_node(3, label="Company", text="公司简介")
        self.G.add_edge(1, 3, relation="belongs_to")
        self.G.add_edge(2, 3, relation="belongs_to")

    def connect_neo4j(self, uri: str, user: str, password: str):
        """Connect to a Neo4j instance using the official driver."""
        if GraphDatabase is None:
            raise RuntimeError("neo4j driver not installed; add 'neo4j' to requirements.txt")
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def kg_insert_nodes_edges(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
        """Insert nodes and edges into Neo4j.

        nodes: list of {"id": <str|int>, "labels": ["Label"], "props": {...}}
        edges: list of {"from": id, "to": id, "rel": "REL", "props": {...}}

        Uses MERGE semantics so repeated runs are idempotent.
        If Neo4j is not connected, falls back to updating the in-memory NetworkX graph.
        """
        if not self.driver:
            # Fallback: update NetworkX
            for n in nodes:
                nid = n.get("id")
                props = n.get("props", {})
                label = ",".join(n.get("labels", [])) or None
                self.G.add_node(nid, label=label, **props)
            for e in edges:
                self.G.add_edge(e.get("from"), e.get("to"), **e.get("props", {}), relation=e.get("rel"))
            return

        def _create(tx, node):
            labels = ":".join(node.get("labels", [])) or ""
            props = node.get("props", {})
            # ensure unique identifier property `kg_id` if provided, otherwise use `id` value
            kg_id = node.get("id")
            props_with_id = {**props, "kg_id": kg_id}
            props_str = ", ".join([f"{k}: $props.{k}" for k in props_with_id.keys()])
            cypher = f"MERGE (n:{labels} {{kg_id: $kg_id}}) SET n += $props"
            tx.run(cypher, kg_id=kg_id, props=props)

        with self.driver.session() as session:
            for node in nodes:
                session.write_transaction(_create, node)

            def _create_rel(tx, e):
                cy = (
                    "MATCH (a {kg_id: $from_id}), (b {kg_id: $to_id}) "
                    "MERGE (a)-[r:" + e.get("rel", "RELATED") + "]->(b) SET r += $props"
                )
                tx.run(cy, from_id=e.get("from"), to_id=e.get("to"), props=e.get("props", {}))

            for e in edges:
                session.write_transaction(_create_rel, e)

    def search_entities(self, query: str, topk: int = 10) -> List[Tuple[int, float, str]]:
        """Search entities by `text` or `label` property.

        Returns list of tuples: (node_identifier, score, 'kg')
        """
        results = []
        if not self.driver:
            # simple NetworkX match
            for n, d in self.G.nodes(data=True):
                txt = d.get("text", "") or ""
                lbl = d.get("label", "") or ""
                if query.lower() in txt.lower() or query.lower() in lbl.lower():
                    results.append((n, 1.0, "kg"))
            return results[:topk]

        cypher = (
            "MATCH (n) WHERE (exists(n.text) AND toLower(n.text) CONTAINS toLower($q))"
            " OR (exists(n.label) AND toLower(n.label) CONTAINS toLower($q)) "
            "RETURN n.kg_id as kg_id, n.text as text LIMIT $limit"
        )
        with self.driver.session() as session:
            res = session.run(cypher, q=query, limit=topk)
            for r in res:
                kg_id = r.get("kg_id")
                results.append((kg_id, 1.0, "kg"))
        return results

    def expand_neighbors(self, node_id: int, hops: int = 1) -> List[Any]:
        if not self.driver:
            return list(nx.single_source_shortest_path_length(self.G, node_id, cutoff=hops).keys())

        # Use kg_id property to find node
        cy = (
            "MATCH (s {kg_id: $kgid})-[*1..$hops]-(m) "
            "RETURN DISTINCT m.kg_id as kg_id LIMIT 1000"
        )
        with self.driver.session() as session:
            res = session.run(cy, kgid=node_id, hops=hops)
            return [r.get("kg_id") for r in res]
