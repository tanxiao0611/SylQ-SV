"""A library of helper functions for working with the PySlang AST."""
import pyslang as ps

def get_module_name(module) -> str:
    """From module syntax object return the module name."""
    return module.header.name.value

class SlangSymbolVisitor:
    """Visits a Slang AST by each Symbol."""

    def __init__(self):
        self.symbol_id_to_symbol = dict()
        self.sourceRange_to_symbol_id = dict()
        self.kind_to_symbol_id = dict()

        self.symbol_id = 0
    
    def visit(self, symbol):
        print(f"visiting {symbol}")
        self.symbol_id_to_symbol[self.symbol_id] = symbol
        
        try:
            self.kind_to_symbol_id[symbol.kind].append(self.symbol_id)
        except KeyError:
            self.kind_to_symbol_id[symbol.kind] = [self.symbol_id]

        try:
            hashable_sourceRange = (symbol.syntax.sourceRange.start, symbol.syntax.sourceRange.end)
            try:
                self.sourceRange_to_symbol_id[hashable_sourceRange].append(self.symbol_id)
            except KeyError:
                self.sourceRange_to_symbol_id[hashable_sourceRange] = [self.symbol_id]
        except AttributeError:
            pass
        self.symbol_id += 1


class SlangNodeVisitor:
    """Visits a Slang AST by each Node."""
    visitor_for_symbol = None
    
    def __init__(self, visitor_for_symbol):
        print("building a node visitor")
        self.visitor_for_symbol = visitor_for_symbol
        self.node_id_to_node = dict()
        self.node_id_to_pid  = {0:None}
        self.node_id_to_cids = dict()
        self.node_id_to_cids = {0:None}
        self.node_id_to_name = {0:""}
        self.node_id_to_name_symbol = {0:""}
        self.node_id_to_predicates = {0:[]}

        self.kind_to_node_ids = dict()
        
        self.level = 0
        self.num_children_in_level = 1
        self.num_children_in_next_level = 0
        self.num_children_processed = 0
        self.processed_children_ids = list()

        self.node_id = 0

    def traverse_tree(self, starting_node):
        """Traverse the AST."""

        self.queue = [starting_node]

        while len(self.queue) > 0:
            curr_node = self.queue.pop(0)
            self.visit(curr_node, use_queue=True)
        
        return True

    def process_node_for_predicates(self, node):
        pid = self.node_id_to_pid[self.node_id]
        if pid == None:
            return
        p_node = self.node_id_to_node[pid]

        new_predicate = []
        if p_node.kind == ps.SyntaxKind.ConditionalStatement:
            if p_node.statement == node:
                new_predicate = [(self.find_corresponding_child_id(pid, p_node.predicate), True)]
            elif p_node.elseClause == node:
                new_predicate = [(self.find_corresponding_child_id(pid, p_node.predicate), False)]
        if p_node.kind == ps.SyntaxKind.ConditionalExpression:
            if   p_node.left  == node:
                new_predicate = [(self.find_corresponding_child_id(pid, p_node.predicate), True)]
            elif p_node.right == node:
                new_predicate = [(self.find_corresponding_child_id(pid, p_node.predicate), False)]
        
        prev_predicates = self.node_id_to_predicates[pid]
        self.node_id_to_predicates[self.node_id] = prev_predicates+new_predicate

    def process_node_for_name(self, node):
        pid = self.node_id_to_pid[self.node_id]
        if pid == None:
            return
        p_node = self.node_id_to_node[pid]
        prev_name = self.node_id_to_name[pid]

        new_name = None
        # if node.kind == ps.SyntaxKind.CompilationUnit:
        #     new_name = "/"
        print(f"kind {node.kind}")
        if node.kind == ps.SyntaxKind.ModuleDeclaration:
            new_name = node.header.name.value
        elif node.kind == ps.SyntaxKind.HierarchicalInstance:
            new_name = node.decl.name.value
        elif node.kind == ps.SyntaxKind.Declarator:
            new_name = node.name.value

        if new_name == None:
            self.node_id_to_name[self.node_id] = f"{prev_name}"
        else:
            self.node_id_to_name[self.node_id] = f"{prev_name}.{new_name}"

    def extract_kinds_from_descendants(nid, desired_kinds=[ps.TokenKind.Identifier]):
        desired_nids = list()

        queue = [(nid, [])]
        while len(queue) > 0:
            curr_nid, curr_metadata = queue.pop(0)
            curr_node = self.node_id_to_node[curr_nid]
            
            if curr_node.kind in desired_kinds:
                desired_nids.append(curr_nid)

            for curr_cid in self.node_id_to_cids[curr_nid]:
                new_metadata = []
                queue.append((curr_cid, curr_metadata+new_metadata))

        return desired_nids

    def visit(self, node, use_queue=True):
        self.node_id_to_node[self.node_id] = node
        self.node_id_to_cids[self.node_id] = list()

        #self.process_node_for_predicates(node)
        self.process_node_for_name(node)
        try:
            self.kind_to_node_ids[node.kind].append(self.node_id)
        except KeyError:
            self.kind_to_node_ids[node.kind] = [self.node_id]

        self.num_children_processed += 1
        self.processed_children_ids.append(self.node_id)
        try:
            for i in range(len(node)):
                self.num_children_in_next_level += 1
                child_id = self.node_id + (self.num_children_in_level - self.num_children_processed) + self.num_children_in_next_level
                self.node_id_to_pid[child_id] = self.node_id
                self.node_id_to_cids[self.node_id].append(child_id)

                if use_queue:
                    self.queue.append(node.__getitem__(i))
        except TypeError:
            # This exception is required because, unlike SyntaxNode, Token
            # objects do not have a len() function (i.e., getChildCount)
            pass
        

        if self.num_children_processed == self.num_children_in_level:
            self.level += 1
            self.num_children_in_level = self.num_children_in_next_level
            self.num_children_in_next_level = 0
            self.num_children_processed = 0
            self.processed_children_ids = list()

        self.node_id += 1