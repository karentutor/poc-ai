jq 'to_entries
    | map({ (.key): ( .value[0]                         # first hit
                      | split(":")[0]                   # keep textId only
                )})
    | add' matched.json > field_values.json