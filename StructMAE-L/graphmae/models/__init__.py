from .edcoder import PreModel


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate


    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features

    sc_type = args.sc_type
    sc_type2 = args.sc_type2
    sc_num_layers1 = args.sc_num_layers1
    sc_num_layers2 = args.sc_num_layers2

    alpha = args.alpha
    alpha_sc2 = args.alpha_sc2
    curMode = args.curMode
    sc_sigmoid = args.sc_sigmoid

    model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        sc_type=sc_type,
        sc_type2=sc_type2,
        sc_num_layers1=sc_num_layers1,
        sc_num_layers2=sc_num_layers2,
        alpha = alpha,
        alpha_sc2 = alpha_sc2,
        sc_sigmoid = sc_sigmoid
    )
    return model
