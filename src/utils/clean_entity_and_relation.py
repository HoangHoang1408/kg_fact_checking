import re


def clean_original_entity(entity):
    entity = entity.replace('"', "").replace("_", " ")
    entity = " ".join(entity.split())
    return entity


def clean_orignal_relation(relation):
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", relation)
    # Convert to lowercase
    return spaced.lower()
