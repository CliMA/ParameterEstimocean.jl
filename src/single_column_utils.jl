function set!(model::Oceananigans.AbstractModel,
              td::TruthData, i)

    set!(model, b = td.b[i],
                u = td.u[i],
                v = td.v[i],
                e = td.e[i]
        )

    model.clock.time = td.t[i]

    return nothing
end

set!(pm::ParameterizedModel, td::TruthData, i) = set!(pm.model, td, i)
