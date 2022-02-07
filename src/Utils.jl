module Utils

field_name_pairs(value, field_names, args...) = NamedTuple(name => value for name in field_names)
field_name_pairs(t::Union{Tuple, AbstractArray}, field_names, args...) =
    NamedTuple(name => t[i] for (i, name) in enumerate(field_names))

function field_name_pairs(nt::Union{NamedTuple, Dict}, field_names, nt_name="")
    nt_field_names = keys(nt)
    nt_summary = summary(nt)

    # Validate user-supplied NamedTuple
    all(name ∈ nt_field_names for name in field_names) ||
        throw(ArgumentError("$nt_name $nt_summary must have values for every field in $field_names " *
                            " but only has values for $nt_field_names"))

    return nt
end

function prettyvector(v::AbstractVector, bookends=3)
    separator = " … "
    beginning = [string(v[i]) for i=1:bookends]
    ending = [string(v[end+1-i]) for i=1:bookends]

    for i = 1:bookends-1
        beginning[i] *= " "
        ending[i] *= " "
    end

    N = length(v)

    return string("[", beginning..., separator, ending..., "] ($N elements)")
end

end # module
