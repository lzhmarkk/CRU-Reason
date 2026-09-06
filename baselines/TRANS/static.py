feats_to_nodes = {
    'conditions': 'co',
    'procedures': 'pr',
    'drugs': 'dh',
    'co': 'conditions',
    'pr': 'procedures',
    'dh': 'drugs'
}

graph_meta = (['visit', 'co', 'pr', 'dh'],
              [('co', 'in', 'visit'),
               ('pr', 'in', 'visit'),
               ('dh', 'in', 'visit'),
               ('visit', 'connect', 'visit'),
               ('visit', 'has', 'co'),
               ('visit', 'has', 'pr'),
               ('visit', 'has', 'dh')])
