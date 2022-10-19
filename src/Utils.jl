module Utils

using CUDA

tupleit(t) = try
    Tuple(t)
catch
    tuple(t)
end

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
    N = length(v)

    if N < 2bookends + 4
        content = string.(v) .* ", "
        content[end] = content[end][1:end-2]
        return string("[", content..., "]")
    else
        separator = " … "
        beginning = [string(v[i]) for i=1:bookends]
        ending = [string(v[end+1-i]) for i=1:bookends]

        for i = 1:bookends-1
            beginning[i] *= ", "
            ending[i] *= ", "
        end

        N = length(v)

        return string("[", beginning..., separator, ending..., "] ($N elements)")
    end
end

"""
    map_gpu_to_rank(; comm = MPI.COMM_WORLD)   

maps one rank to one GPU by leveraging the CUDA.device!(d::Int) function. 
"""
function map_gpus_to_ranks!(; comm = MPI.COMM_WORLD)   
    rank = MPI.Comm_rank(comm)
    name = MPI.Get_processor_name()
    hash = name_to_hash(node_name)

    node_comm =  MPI.Comm_split(comm, hash, nrank)
    node_rank =  MPI.Comm_rank(node_rank)
    # Check that there are enough GPUs to ranks in every node
    node_rank > length(CUDA.devices()) - 1 && 
        throw(ArgumentError("Not enough GPUs per ranks in a node. Reduce the number of processes per node"))
    CUDA.device!(node_rank)
end

function name_to_hash(node)
    hash = Int64(0)
    for i=1:length(node)
        hash = hash + (Int(node[i])+1)*10^i
    end
    return hash
end 

end # module
