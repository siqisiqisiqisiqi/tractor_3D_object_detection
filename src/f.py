
def combine_dicts_to_list(dict1, dict2):
    combined_dict = {}

    for key in sorted(set(dict1) | set(dict2)):  # Union of keys from both dictionaries
        combined_dict[key] = []
        if key in dict1:
            if isinstance(dict1[key], list):
                combined_dict[key].extend(dict1[key])
            else:
                combined_dict[key].append(dict1[key])
        if key in dict2:
            if isinstance(dict2[key], list):
                combined_dict[key].extend(dict2[key])
            else:
                combined_dict[key].append(dict2[key])

    return combined_dict
