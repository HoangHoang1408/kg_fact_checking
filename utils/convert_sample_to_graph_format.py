import random


def deduplicate_triplets(triplets):
    connections = set()
    filtered_triplets = []
    for triplet in triplets:
        connection = tuple(sorted([triplet[0], triplet[2]]))
        if connection not in connections:
            connections.add(connection)
            filtered_triplets.append(triplet)
    return filtered_triplets


def convert_to_triplet_format(sample, unknown_prefix="unknown_entity"):
    def revert_relation(relation):
        new_relation = []
        for r in relation[::-1]:
            if r.startswith("~"):
                new_relation.append(r[1:])
            else:
                new_relation.append("~" + r)
        return tuple(new_relation)

    relation_to_entity = {}
    for entity, relations in sample["Evidence"].items():
        for relation in relations:
            relation = tuple(relation)
            relation_to_entity[relation] = entity

    triplets = []
    unknown_id = 0
    for entity, relations in list(sample["Evidence"].items())[::-1]:
        for org_relation in relations:
            relation = revert_relation(tuple(org_relation))
            if relation in relation_to_entity:
                triplets.append((relation_to_entity[relation], relation, entity))
            else:
                triplets.append(
                    (entity, org_relation, f"{unknown_prefix}_{unknown_id}")
                )
                unknown_id += 1

    # split relations
    splitted_triplets = []
    for triplet in deduplicate_triplets(triplets):
        head, relation, tail = triplet
        if len(relation) > 1:
            for i in range(len(relation)):
                if i == 0:
                    splitted_triplets.append(
                        (head, relation[i], f"{unknown_prefix}_{unknown_id}")
                    )
                elif i == len(relation) - 1:
                    splitted_triplets.append(
                        (f"{unknown_prefix}_{unknown_id}", relation[i], tail)
                    )
                else:
                    splitted_triplets.append(
                        (
                            f"{unknown_prefix}_{unknown_id}",
                            relation[i],
                            f"{unknown_prefix}_{unknown_id + 1}",
                        )
                    )
                    unknown_id += 1
        else:
            splitted_triplets.append((head, relation[0], tail))
        unknown_id += 1

    # check unknown entities
    unknown_entity_mapping = {}
    for i, e in enumerate(sample["present"]["out_entities"]):
        unknown_entity_mapping[e] = f"{unknown_prefix}_{unknown_id + i}"
    unknown_id += len(unknown_entity_mapping)

    for i in range(len(splitted_triplets)):
        triplet = list(splitted_triplets[i])
        for j in [0, 2]:
            if triplet[j] in unknown_entity_mapping:
                triplet[j] = unknown_entity_mapping[triplet[j]]
        splitted_triplets[i] = tuple(triplet)

    return splitted_triplets


def augment_triplet_format(triplets):
    def revert_relation(relation):
        if "~" in relation:
            return relation[1:]
        return "~" + relation

    random.shuffle(triplets)
    augmented_triplets = []
    for triplet in triplets:
        head, relation, tail = triplet
        augmented_triplets.append(
            random.choice([(tail, revert_relation(relation), head), triplet])
        )
    return augmented_triplets


def remap_unknown_entities(triplets, unknown_prefix="unknown_entity"):
    unknown_entity_map = {}
    next_unknown_id = 0
    remapped_triplets = []
    for triplet in triplets:
        remapped_triplet = []
        for item in triplet:
            if item.startswith(unknown_prefix):
                if item not in unknown_entity_map:
                    unknown_entity_map[item] = f"{unknown_prefix}_{next_unknown_id}"
                    next_unknown_id += 1
                remapped_triplet.append(unknown_entity_map[item])
            else:
                remapped_triplet.append(item)
        remapped_triplets.append(tuple(remapped_triplet))
    return remapped_triplets


def deduplicate_unknown_entity(triplets):
    def merge_sets(sets):
        merged = []

        for current_set in sets:
            merged_with_existing = False
            for merged_set in merged:
                if current_set & merged_set:
                    merged_set.update(current_set)
                    merged_with_existing = True
                    break
            if not merged_with_existing:
                merged.append(set(current_set))

        while True:
            changes = False
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if merged[i] & merged[j]:
                        merged[i].update(merged[j])
                        del merged[j]
                        changes = True
                        break
                if changes:
                    break
            if not changes:
                break

        return merged

    def revert_relation(relation):
        if relation.startswith("~"):
            return relation[1:]
        else:
            return "~" + relation

    triplet_mapping = {}
    for triplet in triplets:
        if triplet[0].startswith("unknown_entity"):
            key = (triplet[2], revert_relation(triplet[1]))
            triplet_mapping[key] = triplet_mapping.get(key, set()) | {triplet[0]}

        elif triplet[2].startswith("unknown_entity"):
            key = (triplet[0], triplet[1])
            triplet_mapping[key] = triplet_mapping.get(key, set()) | {triplet[2]}

    groups = merge_sets(triplet_mapping.values())
    mapping = {}
    for group in groups:
        key = list(group)[0]
        for entity in group:
            mapping[entity] = key

    for i in range(len(triplets)):
        triplet = triplets[i]
        if triplet[0].startswith("unknown_entity"):
            triplets[i] = (mapping[triplet[0]], triplet[1], triplet[2])
        elif triplet[2].startswith("unknown_entity"):
            triplets[i] = (triplet[0], triplet[1], mapping[triplet[2]])

    return deduplicate_triplets(triplets)


def convert_sample_to_graph_format(sample):
    triplets = convert_to_triplet_format(sample)
    triplets = deduplicate_unknown_entity(triplets)
    triplets = remap_unknown_entities(triplets)
    triplets = augment_triplet_format(triplets)

    text = ""
    for triplet in triplets:
        text += f"{triplet[0]} || {triplet[1]} || {triplet[2]}\n"
    return text.strip()
