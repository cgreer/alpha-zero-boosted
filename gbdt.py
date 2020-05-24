'''
Check out ./sample_lgbm.json for sample data
'''
import sys
import typing
from collections import defaultdict, Counter
from dataclasses import dataclass
import json


@dataclass
class SplitDecision:
    feature: int
    threshold: float
    rule: str

    def __str__(self):
        return f"{self.feature} {self.rule} {self.threshold}"

    def binarize(self):
        if 0 <= self.threshold <= 1:
            if self.rule == ">":
                return (self.feature, 1)
            else:
                return (self.feature, 0)
        else:
            return None

    def binarize_str(self):
        if 0 <= self.threshold <= 1:
            if self.rule == ">":
                return f"{self.feature}=1"
            else:
                return f"{self.feature}=0"
        else:
            return str(self)


def translate_rule(split_decision_type, node_relation):
    if split_decision_type == "<=":
        if node_relation == "left_child":
            return "<="
        else:
            return ">"
    else:
        raise KeyError(f"Unknown split_decision_type: split_decision_type")


@dataclass
class LGBMTree:
    model: typing.Dict

    @classmethod
    def load(cls, model_dump):
        tree = cls(
            model=json.loads(open(model_dump, 'r').read()),
        )
        tree.make_bidirectional_connections()
        return tree

    def iter_tree_root_nodes(self):
        # Yield the root node of each tree
        for tree in self.model["tree_info"]:
            # Skip over meta tree info
            yield tree["tree_structure"]

    def iter_nodes(self, root_node):
        queue = [root_node]
        while queue:
            node = queue.pop()
            yield node
            for node_child in self.node_children(node):
                queue.append(node_child)

    def decision_path(self, node):
        # Decisions to get from root node to this state
        path = []
        current_node = node
        while True:
            if current_node["parent"] is None:
                break
            rule = translate_rule(current_node["parent"]["decision_type"], current_node["parent_relationship"])
            path.append(
                SplitDecision(
                    feature=current_node["parent"]["split_feature"],
                    threshold=current_node["parent"]["threshold"],
                    rule=rule,
                )
            )
            current_node = current_node["parent"]
        return path[::-1]

    def node_children(self, node):
        if node["is_leaf"]:
            return []

        # Connect children to parent
        children = []
        for child in ("left_child", "right_child"):
            if child in node:
                children.append(node[child])
        return children

    def make_bidirectional_connections(self):
        for tree in self.iter_tree_root_nodes():

            tree["parent"] = None

            queue = [tree]
            while queue:
                node = queue.pop()

                if "leaf_index" in node:
                    node["is_leaf"] = True
                    continue
                else:
                    node["is_leaf"] = False

                # Connect children to parent
                for child in ("left_child", "right_child"):
                    if child in node:
                        node[child]["parent"] = node
                        node[child]["parent_relationship"] = child
                        queue.append(node[child])

    def node_depth(self, node):
        current_node = node
        depth = 0
        while True:
            if current_node.parent is None:
                # Reached root
                break
            current_node = current_node.parent
            depth += 1
        return depth

    def code_gen_extra_features(self, patterns):

        def code_gen_if_statement(pat, pat_idx):
            fi1 = pat[0][0]
            fv1 = pat[0][1]
            fi2 = pat[1][0]
            fv2 = pat[1][1]
            s = f"    if (original_features[{fi1}] == {fv1}) and (original_features[{fi2}] == {fv2}):\n"
            s += f"        extra_features[{pat_idx}] = 1\n"
            return s

        python_code = "\ndef generate_extra_features(original_features):\n"
        python_code += f"    extra_features = [0] * {len(patterns)}\n"
        for i, pattern in enumerate(patterns):
            if_statement = code_gen_if_statement(pattern, i)
            python_code += if_statement
        python_code += "    return extra_features\n"
        return python_code

    def decision_pattern_statistics(self, max_length=3, max_tree=None):
        '''
        combination: info
            ((0, 1), (72, 0), (...): count
        '''
        pattern_counts = Counter()
        pattern_weights = defaultdict(float)
        for i, tree_root_node in enumerate(self.iter_tree_root_nodes()):
            if max_tree and (i > max_tree):
                break
            for node in self.iter_nodes(tree_root_node):
                d_path = self.decision_path(node)
                d_path = [x.binarize() for x in d_path]

                for choice_length in range(2, max_length + 1):
                    nary_choices = d_path[:choice_length]
                    if len(nary_choices) == choice_length:
                        nary_choices.sort()
                        pattern_counts[tuple(nary_choices)] += 1
                        pattern_weights[tuple(nary_choices)] += 1 # XXX wire up
        return pattern_counts, pattern_weights


def state_translation(decision_path, binarized=False):
    from connect_four import State, BOARD_POSITIONS

    feature_values = []
    for dec in decision_path:
        if binarized:
            feat, value = dec
        else:
            feat, value = dec.binarize()
        feat = int(feat)
        value = int(value)
        feature_values.append((feat, value))

    as_agent = None
    whose_move = None
    board = [[0] * 6 for _ in range(7)]
    for feature, value in feature_values:
        if feature == 0:
            as_agent = value
            continue
        if feature == 1:
            whose_move = value
            continue
        if feature == 2:
            continue

        # Board position
        fprime = feature - 3
        player = 1
        if fprime >= 42:
            player = 2
            fprime = fprime - 42

        position = BOARD_POSITIONS[fprime]
        pos_value = player if value == 1 else -1
        board[position[0]][position[1]] = pos_value

    state = State(
        board=board,
        whose_move=whose_move,
    )

    return (as_agent, state)


def display_node_info(lgbm_tree, node):
    from connect_four import Environment
    from rich import print as rprint
    env = Environment()

    count = node["leaf_count"] if node["is_leaf"] else node["internal_count"]
    value = node["leaf_value"] if node["is_leaf"] else node["internal_value"]

    decision_path = lgbm_tree.decision_path(node)
    as_agent, state = state_translation(decision_path)
    rprint("AS AGENT", as_agent)
    # rprint("split gain", node["split_gain"])
    rprint("observation counts", count)
    rprint("value", value)
    rprint(env.text_display(state))


def analysis_1(tree_dump_path):
    from connect_four import Environment
    # from rich import print as rprint
    env = Environment() # noqa
    lgbm_tree = LGBMTree.load(tree_dump_path)

    # Show sample pattern
    pattern_counts, pattern_weights = lgbm_tree.decision_pattern_statistics(2, 10)
    for pattern in pattern_counts.most_common():
        print(pattern)
        break

    # Do coverage analysis
    for first_n_trees in (100, 200, 300, 1000):
        print("\nFirst n trees", first_n_trees)
        pattern_counts, pattern_weights = lgbm_tree.decision_pattern_statistics(2, first_n_trees)
        total_binary_patterns = 0
        total_trinary_patterns = 0
        for pattern in pattern_counts.most_common():
            if len(pattern[0]) == 2:
                total_binary_patterns += pattern[1]
            if len(pattern[0]) == 3:
                total_trinary_patterns += pattern[1]
        print("binary patterns", total_binary_patterns)
        print("trinary patterns", total_trinary_patterns)

        print(len(list(lgbm_tree.iter_tree_root_nodes())))
        for top_common in range(20, 300, 20):
            top_common_count = 0
            for i, pattern in enumerate(pattern_counts.most_common()):
                if i > top_common:
                    break
                # print()
                # print(pattern)

                as_agent, state = state_translation(pattern[0], binarized=True)
                # rprint("AS AGENT", as_agent)
                # rprint(env.text_display(state))
                top_common_count += pattern[1]
            print(i, pattern[1], "patterns covered", top_common_count)

    return

    pattern_counts, pattern_weights = lgbm_tree.decision_pattern_statistics(max_length=2, max_tree=100)
    patterns = [x[0] for x in pattern_counts.most_common(100)]
    autogen_code = lgbm_tree.code_gen_extra_features(patterns)
    print(autogen_code)
    with open('./extra_features.py', 'w') as f:
        f.write(autogen_code)

    return

    for tree_num, tree_root_node in enumerate(lgbm_tree.iter_tree_root_nodes()):
        if tree_num != 0:
            continue

        for node in lgbm_tree.iter_nodes(tree_root_node):
            dec_path = lgbm_tree.decision_path(node)
            if len(dec_path) != 3:
                continue
            print()
            print(" AND ".join([x.binarize_str() for x in lgbm_tree.decision_path(node)]))
            display_node_info(lgbm_tree, node)
        break
    pass


def retrain_experiment(tree_dump_path, stashed_training_base):
    from intuition_model import GBDTValue
    import numpy

    print("Loading tree")
    lgbm_tree = LGBMTree.load(tree_dump_path)

    # Autogenerate extra feature generation
    print("autogen extra feature gen")
    pattern_counts, _ = lgbm_tree.decision_pattern_statistics(max_length=2, max_tree=200)
    patterns = [x[0] for x in pattern_counts.most_common(100)]
    autogen_code = lgbm_tree.code_gen_extra_features(patterns)
    with open('./extra_features.py', 'w') as f:
        f.write(autogen_code)

    # Use autogen_code to retrain using extra features
    from extra_features import generate_extra_features
    print("Loading Original Features")
    train_features, train_labels, test_features, test_labels = GBDTValue.load_stashed_training_data(stashed_training_base)

    print("train", len(train_features))
    print(train_features[:1])

    print("test", len(test_features))
    print(test_features[:1])

    print("Building new features")
    new_train_features = []
    for i, features in enumerate(train_features):
        if i % 100000 == 0:
            print("got through", i)

        new_features = numpy.concatenate(
            (
                features,
                numpy.asarray(
                    generate_extra_features(features)
                )
            )
        )
        new_train_features.append(new_features)
    new_train_features = numpy.asarray(new_train_features)

    new_test_features = []
    for features in test_features:
        new_features = numpy.concatenate(
            (
                features,
                numpy.asarray(
                    generate_extra_features(features)
                )
            )
        )
        new_test_features.append(new_features)
    new_test_features = numpy.asarray(new_test_features)

    print("Training Experimental")
    model = GBDTValue()
    model.train_from_training_data(
        new_train_features, # swap out
        train_labels,
        new_test_features,
        test_labels,
    )


def retrain(stashed_training_base):
    from intuition_model import GBDTValue
    print("Retraining w/ Stashed Data")
    train_features, train_labels, test_features, test_labels = GBDTValue.load_stashed_training_data(stashed_training_base)
    model = GBDTValue()
    model.train_from_training_data(
        train_features,
        train_labels,
        test_features,
        test_labels,
    )


if __name__ == "__main__":
    tree_dump_path = sys.argv[1]
    stashed_training_base = sys.argv[2]
    # analysis_1(tree_dump_path)
    retrain(stashed_training_base)
    retrain_experiment(tree_dump_path, stashed_training_base)
