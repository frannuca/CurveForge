create table quote_keys
(
    quote_id   serial
        primary key,
    symbol     varchar(64)                            not null,
    source     varchar(64)                            not null,
    field      varchar(64)                            not null,
    quote_key  varchar(195) generated always as ((((((symbol)::text || '|'::text) || (source)::text) || '|'::text) ||
                                                  (field)::text)) stored,
    created_at timestamp with time zone default now() not null,
    constraint quote_keys_pk
        unique (symbol, source, field)
);

create table market_data
(
    quote_id      integer                  not null
        constraint fk_market_data_quote
            references quote_keys,
    as_of         timestamp with time zone not null,
    as_of_date    date generated always as (((as_of AT TIME ZONE 'UTC'::text))::date) stored,
    is_closed     boolean                  not null,
    is_settlement boolean                  not null,
    value         double precision         not null
);

create table quant.metrics
(
    quote_id      integer                  not null
        constraint fk_market_data_metrics_quote
            references quant.quote_keys,
    as_of         timestamp with time zone not null,
    as_of_date    date generated always as (((as_of AT TIME ZONE 'UTC'::text))::date) stored,
    metric_name   text                     not null,
    value         double precision         not null
);
