-- ParkVision multicamera Supabase schema draft
-- ============================================
--
-- DRAFT ONLY. Do not run this file against production.
-- This is a review document for the future multicamera migration, not a real
-- migration. It is intentionally additive and does not change the current iOS
-- public contract.
--
-- Public contract tables that must remain consumable by ParkVision-IOS:
--   public.zones              - logical zone catalog
--   public.occupancy          - consolidated zone occupancy
--   public.parking_spaces     - consolidated per-space state for the iOS map
--   public.occupancy_history  - consolidated historical zone occupancy
--
-- Internal multicamera tables proposed in this draft:
--   public.cameras
--   public.physical_spaces
--   public.camera_rois
--   public.space_observations
--
-- Important product rule:
--   public.space_observations is internal sensor data. iOS must not consume
--   raw camera/ROI observations. The backend fusion layer should publish only
--   consolidated results to public.occupancy and public.parking_spaces.
--
-- ML/TVT rule:
--   This schema does not change model training, model architecture, crop size,
--   class semantics, or the current volume-aware crop used as CNN input.
--   Homography/top-down metadata may be added later for mapping or debugging,
--   but it must not replace the volumetric crop as model input.


-- =============================================================================
-- Existing public contract, documented only
-- =============================================================================

-- public.zones remains the logical zone catalog.
-- Existing shape from snapshot:
--   id text primary key
--   name text not null
--   total_spaces integer not null
--   display_order integer default 0

-- public.occupancy remains the consolidated zone-level state consumed by iOS,
-- widget, alerts, and geofence flows.
-- Existing shape from snapshot:
--   zone_id text primary key references public.zones(id)
--   available_spaces integer not null
--   occupied_spaces integer not null
--   confidence double precision default 1.0
--   updated_at timestamptz default now()

-- public.parking_spaces remains the consolidated per-space state consumed by
-- the iOS map. It must represent real physical stalls, not camera-specific ROIs.
-- Compatibility note:
--   During the consolidated publication phase, public.parking_spaces.space_id
--   may map directly to public.physical_spaces.physical_space_id so existing
--   iOS models can keep using space_id without consuming internal tables.
-- Existing shape from snapshot:
--   space_id text primary key
--   zone_id text not null references public.zones(id)
--   layout_name text not null
--   points jsonb not null
--   is_occupied boolean not null default false
--   confidence double precision not null default 0
--   updated_at timestamptz not null default now()

-- public.occupancy_history remains the consolidated historical zone state.
-- It should be populated from fused state, not raw camera observations.


-- =============================================================================
-- Internal table: cameras
-- =============================================================================

create table if not exists public.cameras (
    camera_id text primary key,
    name text not null,
    source text not null,
    enabled boolean not null default true,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now()
);

comment on table public.cameras is
    'Internal multicamera source registry. Not part of the iOS public contract.';

comment on column public.cameras.source is
    'Video source identifier, such as camera index, file path, RTSP URL, or deployment-specific source key.';


-- =============================================================================
-- Internal table: physical_spaces
-- =============================================================================

create table if not exists public.physical_spaces (
    physical_space_id text primary key,
    zone_id text not null references public.zones(id),
    display_name text,
    map_points jsonb,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now(),
    constraint physical_spaces_map_points_is_array
        check (map_points is null or jsonb_typeof(map_points) = 'array')
);

comment on table public.physical_spaces is
    'Internal registry of real parking stalls. One physical space may be observed by multiple camera ROIs.';

comment on column public.physical_spaces.map_points is
    'Optional display/map geometry for the real stall. This is not CNN input.';


-- =============================================================================
-- Internal table: camera_rois
-- =============================================================================

create table if not exists public.camera_rois (
    roi_id text primary key,
    camera_id text not null references public.cameras(camera_id),
    physical_space_id text not null references public.physical_spaces(physical_space_id),
    layout_id text not null,
    zone_id text not null references public.zones(id),
    points jsonb not null,
    enabled boolean not null default true,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now(),
    constraint camera_rois_points_is_array
        check (jsonb_typeof(points) = 'array'),
    constraint camera_rois_points_have_four_vertices
        check (jsonb_array_length(points) = 4)
);

comment on table public.camera_rois is
    'Internal visual ROIs in camera image coordinates. Maps camera-specific polygons to real physical spaces.';

comment on column public.camera_rois.points is
    'Four-point visual polygon in the processed camera image plane. Used for the existing volume-aware crop, not homography CNN input.';


-- =============================================================================
-- Internal table: space_observations
-- =============================================================================

create table if not exists public.space_observations (
    observation_id uuid primary key default gen_random_uuid(),
    camera_id text not null references public.cameras(camera_id),
    roi_id text not null references public.camera_rois(roi_id),
    physical_space_id text not null references public.physical_spaces(physical_space_id),
    zone_id text not null references public.zones(id),
    is_occupied boolean not null,
    confidence double precision not null,
    observed_at timestamp with time zone not null default now(),
    model_name text,
    metadata jsonb not null default '{}'::jsonb,
    constraint space_observations_confidence_range
        check (confidence >= 0 and confidence <= 1)
);

comment on table public.space_observations is
    'Internal raw camera/ROI observations. iOS must not consume this table; backend fusion publishes consolidated state to occupancy and parking_spaces.';

comment on column public.space_observations.confidence is
    'Per-observation model confidence. Zone-level occupancy.confidence remains a consolidated/public value.';


-- =============================================================================
-- Recommended indexes
-- =============================================================================

create index if not exists cameras_enabled_idx
    on public.cameras (enabled);

create index if not exists physical_spaces_zone_id_idx
    on public.physical_spaces (zone_id);

create index if not exists camera_rois_camera_id_idx
    on public.camera_rois (camera_id);

create index if not exists camera_rois_physical_space_id_idx
    on public.camera_rois (physical_space_id);

create index if not exists camera_rois_zone_id_idx
    on public.camera_rois (zone_id);

create index if not exists camera_rois_enabled_idx
    on public.camera_rois (enabled);

create index if not exists space_observations_observed_at_idx
    on public.space_observations (observed_at desc);

create index if not exists space_observations_camera_time_idx
    on public.space_observations (camera_id, observed_at desc);

create index if not exists space_observations_roi_time_idx
    on public.space_observations (roi_id, observed_at desc);

create index if not exists space_observations_physical_space_time_idx
    on public.space_observations (physical_space_id, observed_at desc);

create index if not exists space_observations_zone_time_idx
    on public.space_observations (zone_id, observed_at desc);


-- =============================================================================
-- Draft RLS and policies
-- =============================================================================
--
-- These policies are intentionally conservative for internal multicamera data.
-- Review Supabase auth strategy before turning this draft into a migration.
-- Existing public contract tables keep their current policies unless a separate
-- security review changes them.

alter table public.cameras enable row level security;
alter table public.physical_spaces enable row level security;
alter table public.camera_rois enable row level security;
alter table public.space_observations enable row level security;

-- Optional read access for configuration/debug views. If not needed, omit these.
create policy "Allow public read cameras draft"
    on public.cameras
    for select
    to public
    using (true);

create policy "Allow public read physical spaces draft"
    on public.physical_spaces
    for select
    to public
    using (true);

create policy "Allow public read camera rois draft"
    on public.camera_rois
    for select
    to public
    using (true);

-- Do not grant public select on space_observations by default.
-- iOS must consume public.occupancy and public.parking_spaces only.

create policy "Allow authenticated manage cameras draft"
    on public.cameras
    for all
    to authenticated
    using (true)
    with check (true);

create policy "Allow authenticated manage physical spaces draft"
    on public.physical_spaces
    for all
    to authenticated
    using (true)
    with check (true);

create policy "Allow authenticated manage camera rois draft"
    on public.camera_rois
    for all
    to authenticated
    using (true)
    with check (true);

create policy "Allow authenticated insert observations draft"
    on public.space_observations
    for insert
    to authenticated
    with check (true);

create policy "Allow authenticated read observations draft"
    on public.space_observations
    for select
    to authenticated
    using (true);


-- =============================================================================
-- Publication compatibility notes
-- =============================================================================
--
-- The future fusion layer should:
--   1. Read recent public.space_observations rows by physical_space_id.
--   2. Resolve camera disagreement using the approved freshness/confidence policy.
--   3. Publish one consolidated row per real stall to public.parking_spaces.
--   4. Publish consolidated zone totals to public.occupancy.
--   5. Append public.occupancy_history from consolidated zone state only.
--
-- The iOS app, widget, alerts, geofence, and trends should continue reading:
--   public.zones
--   public.occupancy
--   public.parking_spaces
--   public.occupancy_history
--
-- They should not read:
--   public.space_observations
