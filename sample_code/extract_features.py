# -*- coding: utf-8 -*-
# @Time    : 8/5/19 4:13 PM
# @Author  : zchai
import sys
import json


def get_features(file_path):
    with open(file_path, 'rb') as f:
        data_list = json.load(f)

    domain_value_list = []
    intent_value_list = []
    slots_key_list = []
    slots_value_list = []
    for data in data_list:
        domain_value = data['domain']
        intent_value = data['intent']
        if 'slots' not in data.keys():
            print(data)
            continue
        else:
            slots = data['slots']
            if type(slots) != dict:
                slots = {}
                print(slots)
                continue

        if domain_value not in domain_value_list:
            domain_value_list.append(domain_value)

        if intent_value not in intent_value_list:
            intent_value_list.append(intent_value)

        for key, value in slots.items():
            # domain intent key
            """
            try:
                key = '.'.join(\
                    [domain_value, intent_value, key])
            except:
                print(data)
                #sys.exit(1)
            """
            if key not in slots_key_list:
                slots_key_list.append(key)
            if value not in slots_value_list:
                slots_value_list.append(value)

    return domain_value_list, intent_value_list, slots_key_list, slots_value_list


if __name__ == '__main__':
    domain_vals, intent_vals, slot_keys, slot_vals = \
        get_features('../dataset/train.json')
    
    print('----data summary----', len(domain_vals), \
        len(intent_vals), len(slot_keys))
    print(domain_vals)
    print(intent_vals)
    print(slot_keys)