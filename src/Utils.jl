module Utils

field_named_tuple(value, field_names, args...) = NamedTuple(name => value for name in field_names)
field_named_tuple(t::Tuple, field_names, args...) = NamedTuple(name => t[i] for (i, name) in enumerate(field_names))

function field_named_tuple(nt::NamedTuple, field_names, nt_name="")
    nt_field_names = keys(nt)
    nt_summary = summary(nt)

    # Validate user-supplied NamedTuple
    nt_field_names === field_names ||
        throw(ArgumentError("$nt_name $nt_summary must have values for every field in $field_names " *
                            " but only has values for $nt_field_names"))

    return nt
end

end # module
