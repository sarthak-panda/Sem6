--
-- PostgreSQL database dump
--

-- Dumped from database version 15.10
-- Dumped by pg_dump version 15.10

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: auction_deletion_cleanup(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.auction_deletion_cleanup() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF OLD.is_sold THEN
        DELETE FROM public.player_team
        WHERE player_id = OLD.player_id AND team_id = OLD.team_id AND season_id = OLD.season_id;

        DELETE FROM public.awards
        WHERE player_id = OLD.player_id
            AND match_id IN (
                    SELECT match_id 
                    FROM public.match 
                    WHERE season_id = OLD.season_id
            );

        DELETE FROM public.player_match
        WHERE player_id = OLD.player_id AND team_id = OLD.team_id
            AND match_id IN (
                SELECT match_id 
                FROM public.match 
                WHERE season_id = OLD.season_id
            );

        DELETE FROM public.wickets w--check with revanth
        USING public.balls b
        WHERE (
                (w.player_out_id = OLD.player_id OR w.fielder_id = OLD.player_id)
                OR 
                (
                    w.match_id = b.match_id AND w.innings_num = b.innings_num AND w.over_num = b.over_num AND w.ball_num = b.ball_num
                    AND (b.striker_id = OLD.player_id OR b.non_striker_id = OLD.player_id OR b.bowler_id = OLD.player_id)
                )
            )
            AND w.match_id IN (
                SELECT match_id 
                FROM public.match 
                WHERE season_id = OLD.season_id
            );

        DELETE FROM public.extras e
        USING public.balls b
        WHERE e.match_id = b.match_id AND e.innings_num = b.innings_num AND e.over_num = b.over_num AND e.ball_num = b.ball_num
            AND (b.striker_id = OLD.player_id OR b.non_striker_id = OLD.player_id OR b.bowler_id = OLD.player_id)
            AND b.match_id IN (
                SELECT match_id 
                FROM public.match 
                WHERE season_id = OLD.season_id
            );

        DELETE FROM public.batter_score bs
        USING public.balls b
        WHERE bs.match_id = b.match_id AND bs.innings_num = b.innings_num AND bs.over_num = b.over_num AND bs.ball_num = b.ball_num
            AND (b.striker_id = OLD.player_id OR b.non_striker_id = OLD.player_id OR b.bowler_id = OLD.player_id)
            AND b.match_id IN (
                SELECT match_id 
                FROM public.match 
                WHERE season_id = OLD.season_id
            );

        DELETE FROM public.balls
        WHERE (striker_id = OLD.player_id OR non_striker_id = OLD.player_id OR bowler_id = OLD.player_id)
            AND match_id IN (
                SELECT match_id 
                FROM public.match 
                WHERE season_id = OLD.season_id
            );
    END IF;
    RETURN OLD;
END;
$$;


ALTER FUNCTION public.auction_deletion_cleanup() OWNER TO postgres;

--
-- Name: automatic_insertion_into_player_team(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.automatic_insertion_into_player_team() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.is_sold = TRUE THEN
        --i am assuming we do not need to update because only insert after auction done policy
        INSERT INTO public.player_team VALUES (NEW.player_id,NEW.team_id,NEW.season_id);
    END IF;
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.automatic_insertion_into_player_team() OWNER TO postgres;

--
-- Name: automatic_season_id_generation(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.automatic_season_id_generation() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    --NEW.season_id := 'IPL' || NEW.year;
    NEW.season_id := CONCAT('IPL', NEW.year)::varchar(20);
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.automatic_season_id_generation() OWNER TO postgres;

--
-- Name: check_wicketkeeper_for_stumped(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.check_wicketkeeper_for_stumped() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.kind_out = 'stumped' THEN
        IF NOT EXISTS (
            SELECT 1 FROM public.player_match pm
            WHERE pm.match_id = NEW.match_id
            AND pm.player_id = NEW.fielder_id
            AND pm.role = 'wicketkeeper'
        ) THEN
            RAISE EXCEPTION 'for stumped dismissal, fielder must be a wicketkeeper';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;


ALTER FUNCTION public.check_wicketkeeper_for_stumped() OWNER TO postgres;

--
-- Name: insert_awards_after_match(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.insert_awards_after_match() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    orange_cap_player VARCHAR(20);
    purple_cap_player VARCHAR(20);
BEGIN
    IF NEW.win_type IS NOT NULL AND (OLD.win_type IS DISTINCT FROM NEW.win_type) THEN

        WITH runs_o AS (
            SELECT striker_id, SUM(run_scored) AS total_runs
            FROM public.batter_score
            WHERE match_id = NEW.match_id
            GROUP BY striker_id
        ),
        ranked_runs AS (
            SELECT striker_id
            FROM runs_o
            ORDER BY total_runs DESC, striker_id ASC
            LIMIT 1;
        )
        SELECT striker_id INTO orange_cap_player
        FROM ranked_runs

        WITH wickets_o AS (
            SELECT b.bowler_id, COUNT(*) AS total_wickets
            FROM public.balls b
            JOIN public.wickets w 
                ON b.match_id = w.match_id AND b.innings_num = w.innings_num AND b.over_num = w.over_num AND b.ball_num = w.ball_num
            WHERE b.match_id = NEW.match_id
                AND w.kind_out IN ('bowled', 'caught', 'lbw', 'stumped') -- or should i remove this condition so all accepted ('bowled', 'caught', 'lbw', 'runout', 'stumped', 'hitwicket') --check
            GROUP BY b.bowler_id
        ),
        ranked_wickets AS (
            SELECT bowler_id
            FROM wickets_o
            ORDER BY total_wickets DESC, bowler_id ASC
            LIMIT 1;
        )
        SELECT bowler_id INTO purple_cap_player
        FROM ranked_wickets\
        
        INSERT INTO public.awards (match_id, award_type, player_id)
        VALUES
            (NEW.match_id, 'orange_cap', orange_cap_player),
            (NEW.match_id, 'purple_cap', purple_cap_player);
    END IF;
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.insert_awards_after_match() OWNER TO postgres;

--
-- Name: limit_on_international_players_per_team(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.limit_on_international_players_per_team() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    international_count INT;
BEGIN
    SELECT COUNT(*)
    INTO international_count
    FROM public.player_team pt
    JOIN public.player p ON pt.player_id = p.player_id
    WHERE pt.team_id = NEW.team_id
        AND pt.season_id = NEW.season_id
        AND p.country_name <> 'India';
    -- Check if the new player being added is an international player
    IF (SELECT country_name FROM public.player WHERE player_id = NEW.player_id) <> 'India' THEN
        IF international_count >= 3 THEN
            RAISE EXCEPTION 'there could be atmost 3 international players per team per season';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;


ALTER FUNCTION public.limit_on_international_players_per_team() OWNER TO postgres;

--
-- Name: limit_on_number_of_home_matches(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.limit_on_number_of_home_matches() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    home_region_team1 VARCHAR(20);
    home_region_team2 VARCHAR(20);
BEGIN
    SELECT region INTO home_region_team1 FROM public.team WHERE team_id = NEW.team_1_id;
    SELECT region INTO home_region_team2 FROM public.team WHERE team_id = NEW.team_2_id;
    -- League match must be played at home ground of one of the teams
    IF NEW.match_type = 'league' THEN
        IF NEW.venue <> home_region_team1 AND NEW.venue <> home_region_team2 THEN
            RAISE EXCEPTION 'league match must be played at home ground of one of the teams';
        END IF;
        IF NEW.venue = home_region_team1 THEN
            IF EXISTS (
                SELECT 1
                FROM public.match AS m
                WHERE m.match_type = 'league'
                    AND m.season_id = NEW.season_id
                    AND m.venue = home_region_team1
                    AND (m.team_1_id = NEW.team_1_id OR m.team_2_id = NEW.team_1_id)
                    AND (m.team_1_id = NEW.team_2_id OR m.team_2_id = NEW.team_2_id)
            ) THEN
                RAISE EXCEPTION 'each team can play only one home match in a league against another team';
            END IF;
        END IF;
        IF NEW.venue = home_region_team2 THEN
            IF EXISTS (
                SELECT 1
                FROM public.match AS m
                WHERE m.match_type = 'league'
                    AND m.season_id = NEW.season_id
                    AND m.venue = home_region_team2
                    AND (m.team_1_id = NEW.team_2_id OR m.team_2_id = NEW.team_2_id)
                    AND (m.team_1_id = NEW.team_1_id OR m.team_2_id = NEW.team_1_id)
            ) THEN
                RAISE EXCEPTION 'each team can play only one home match in a league against another team';
            END IF;
        END IF;
    END IF;

    RETURN NEW;
END;
$$;


ALTER FUNCTION public.limit_on_number_of_home_matches() OWNER TO postgres;

--
-- Name: match_deletion_cleanup(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.match_deletion_cleanup() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    DELETE FROM public.balls
    WHERE match_id = OLD.match_id;
    
    DELETE FROM public.batter_score
    WHERE match_id = OLD.match_id;
    
    DELETE FROM public.extras
    WHERE match_id = OLD.match_id;
    
    DELETE FROM public.wickets
    WHERE match_id = OLD.match_id;
    
    DELETE FROM public.player_match
    WHERE match_id = OLD.match_id;

    RETURN OLD;
END;
$$;


ALTER FUNCTION public.match_deletion_cleanup() OWNER TO postgres;

--
-- Name: match_id_validation(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.match_id_validation() RETURNS trigger
    LANGUAGE plpgsql
    AS $_$
DECLARE
    season_part VARCHAR(20);
    seq_part INTEGER;
    prev_match_id VARCHAR(20);
BEGIN
    IF NEW.match_id IS NULL THEN
        RAISE EXCEPTION 'null';--check--changed
    END IF;
    IF NEW.match_id !~ '^[a-zA-Z0-9]+[0-9]{3}$' THEN
        RAISE EXCEPTION 'sequence of match id violated';
    END IF;
    season_part := LEFT(NEW.match_id, LENGTH(NEW.match_id) - 3);
    seq_part := CAST(RIGHT(NEW.match_id, 3) AS INTEGER);
    IF NEW.season_id <> season_part THEN
        RAISE EXCEPTION 'sequence of match id violated';
    END IF;
    IF seq_part = 1 THEN
        IF EXISTS (SELECT 1 FROM public.match WHERE match_id = NEW.match_id) THEN
            RAISE EXCEPTION 'sequence of match id violated';
        END IF;
    ELSE
        prev_match_id := season_part || LPAD((seq_part - 1)::TEXT, 3, '0');
        IF NOT EXISTS (SELECT 1 FROM public.match WHERE match_id = prev_match_id) THEN
            RAISE EXCEPTION 'sequence of match id violated';
        END IF;
        IF EXISTS (SELECT 1 FROM public.match WHERE match_id = NEW.match_id) THEN
            RAISE EXCEPTION 'sequence of match id violated';
        END IF;
    END IF;
    RETURN NEW;
END;
$_$;


ALTER FUNCTION public.match_id_validation() OWNER TO postgres;

--
-- Name: season_deletion_cleanup(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.season_deletion_cleanup() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN

    DELETE FROM public.auction
    WHERE season_id = OLD.season_id;

    DELETE FROM public.awards
    WHERE match_id IN (
        SELECT match_id FROM public.match WHERE season_id = OLD.season_id
    );

    DELETE FROM public.balls
    WHERE match_id IN (
        SELECT match_id FROM public.match WHERE season_id = OLD.season_id
    );

    DELETE FROM public.batter_score
    WHERE match_id IN (
        SELECT match_id FROM public.match WHERE season_id = OLD.season_id
    );
    
    DELETE FROM public.extras
    WHERE match_id IN (
        SELECT match_id FROM public.match WHERE season_id = OLD.season_id
    );
    
    DELETE FROM public.wickets
    WHERE match_id IN (
        SELECT match_id FROM public.match WHERE season_id = OLD.season_id
    );

    DELETE FROM public.player_match
    WHERE match_id IN (
        SELECT match_id FROM public.match WHERE season_id = OLD.season_id
    );

    DELETE FROM public.match
    WHERE season_id = OLD.season_id;

    DELETE FROM public.player_team
    WHERE season_id = OLD.season_id;

    RETURN OLD;
END;
$$;


ALTER FUNCTION public.season_deletion_cleanup() OWNER TO postgres;

--
-- Name: set_winner_team_id(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.set_winner_team_id() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    batting_first_team VARCHAR(20);
    batting_second_team VARCHAR(20);
BEGIN
    IF NEW.win_type IS NOT NULL AND (OLD.win_type IS DISTINCT FROM NEW.win_type) THEN
        IF NEW.win_type = 'draw' THEN
            NEW.winner_team_id := NULL;
        ELSE
            IF NEW.toss_winner = 1 THEN
                IF NEW.toss_decide = 'bat' THEN
                    batting_first_team := NEW.team_1_id;
                    batting_second_team := NEW.team_2_id;
                ELSE
                    batting_first_team := NEW.team_2_id;
                    batting_second_team := NEW.team_1_id;
                END IF;
            ELSE
                IF NEW.toss_decide = 'bat' THEN
                    batting_first_team := NEW.team_2_id;
                    batting_second_team := NEW.team_1_id;
                ELSE
                    batting_first_team := NEW.team_1_id;
                    batting_second_team := NEW.team_2_id;
                END IF;
            END IF;

            IF NEW.win_type = 'runs' THEN
                NEW.winner_team_id := batting_first_team;
            ELSIF NEW.win_type = 'wickets' THEN
                NEW.winner_team_id := batting_second_team;
            END IF;        
        END IF;
    END IF;
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.set_winner_team_id() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: auction; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.auction (
    auction_id character varying(20) NOT NULL,
    season_id character varying(20) NOT NULL,
    player_id character varying(20) NOT NULL,
    base_price bigint NOT NULL,
    sold_price bigint,
    is_sold boolean NOT NULL,
    team_id character varying(20) NOT NULL,
    CONSTRAINT auction_base_price_check CHECK ((base_price >= 1000000)),
    CONSTRAINT auction_check CHECK ((((is_sold = false) AND (sold_price IS NULL)) OR ((is_sold = true) AND (sold_price IS NOT NULL) AND (team_id IS NOT NULL) AND (sold_price >= base_price))))
);


ALTER TABLE public.auction OWNER TO postgres;

--
-- Name: awards; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.awards (
    match_id character varying(20) NOT NULL,
    award_type character varying(20) NOT NULL,
    player_id character varying(20) NOT NULL,
    CONSTRAINT awards_award_type_check CHECK (((award_type)::text = ANY ((ARRAY['orange_cap'::character varying, 'purple_cap'::character varying])::text[])))
);


ALTER TABLE public.awards OWNER TO postgres;

--
-- Name: balls; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.balls (
    match_id character varying(20) NOT NULL,
    innings_num smallint NOT NULL,
    over_num smallint NOT NULL,
    ball_num smallint NOT NULL,
    striker_id character varying(20) NOT NULL,
    non_striker_id character varying(20) NOT NULL,
    bowler_id character varying(20) NOT NULL
);


ALTER TABLE public.balls OWNER TO postgres;

--
-- Name: batter_score; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.batter_score (
    match_id character varying(20) NOT NULL,
    over_num smallint NOT NULL,
    innings_num smallint NOT NULL,
    ball_num smallint NOT NULL,
    run_scored smallint NOT NULL,
    type_run character varying(20),
    CONSTRAINT batter_score_run_scored_check CHECK ((run_scored >= 0)),
    CONSTRAINT batter_score_type_run_check CHECK (((type_run)::text = ANY ((ARRAY['running'::character varying, 'boundary'::character varying])::text[])))
);


ALTER TABLE public.batter_score OWNER TO postgres;

--
-- Name: extras; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.extras (
    match_id character varying(20) NOT NULL,
    innings_num smallint NOT NULL,
    over_num smallint NOT NULL,
    ball_num smallint NOT NULL,
    extra_runs smallint NOT NULL,
    extra_type character varying(20) NOT NULL,
    CONSTRAINT extras_extra_runs_check CHECK ((extra_runs >= 0)),
    CONSTRAINT extras_extra_type_check CHECK (((extra_type)::text = ANY ((ARRAY['no_ball'::character varying, 'wide'::character varying, 'byes'::character varying, 'legbyes'::character varying])::text[])))
);


ALTER TABLE public.extras OWNER TO postgres;

--
-- Name: wickets; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.wickets (
    match_id character varying(20) NOT NULL,
    innings_num smallint NOT NULL,
    over_num smallint NOT NULL,
    ball_num smallint NOT NULL,
    player_out_id character varying(20) NOT NULL,
    kind_out character varying(20) NOT NULL,
    fielder_id character varying(20),
    CONSTRAINT wickets_check CHECK (((((kind_out)::text = ANY ((ARRAY['caught'::character varying, 'runout'::character varying, 'stumped'::character varying])::text[])) AND (fielder_id IS NOT NULL)) OR (((kind_out)::text <> ALL ((ARRAY['caught'::character varying, 'runout'::character varying, 'stumped'::character varying])::text[])) AND (fielder_id IS NULL)))),
    CONSTRAINT wickets_kind_out_check CHECK (((kind_out)::text = ANY ((ARRAY['bowled'::character varying, 'caught'::character varying, 'lbw'::character varying, 'runout'::character varying, 'stumped'::character varying, 'hitwicket'::character varying])::text[])))
);


ALTER TABLE public.wickets OWNER TO postgres;

--
-- Name: batter_stats; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.batter_stats AS
 WITH innings_agg AS (
         SELECT b.striker_id AS player_id,
            b.match_id,
            b.innings_num,
            COALESCE(sum(bs.run_scored), (0)::bigint) AS runs_in_innings,
            count(*) FILTER (WHERE (e.match_id IS NULL)) AS balls_faced_in_innings,
            sum(
                CASE
                    WHEN (((bs.type_run)::text = 'boundary'::text) AND (bs.run_scored = ANY (ARRAY[4, 6]))) THEN 1
                    ELSE 0
                END) AS boundary_hits,
                CASE
                    WHEN (max((w.player_out_id)::text) = (b.striker_id)::text) THEN 1
                    ELSE 0
                END AS is_out
           FROM (((public.balls b
             LEFT JOIN public.batter_score bs ON ((((b.match_id)::text = (bs.match_id)::text) AND (b.innings_num = bs.innings_num) AND (b.over_num = bs.over_num) AND (b.ball_num = bs.ball_num))))
             LEFT JOIN public.extras e ON ((((b.match_id)::text = (e.match_id)::text) AND (b.innings_num = e.innings_num) AND (b.over_num = e.over_num) AND (b.ball_num = e.ball_num))))
             LEFT JOIN public.wickets w ON ((((b.match_id)::text = (w.match_id)::text) AND (b.innings_num = w.innings_num) AND (b.over_num = w.over_num) AND (b.ball_num = w.ball_num) AND ((w.player_out_id)::text = (b.striker_id)::text))))
          GROUP BY b.striker_id, b.match_id, b.innings_num
        )
 SELECT ia.player_id,
    (count(DISTINCT ia.match_id))::smallint AS "Mat",
    (count(*))::smallint AS "Inns",
    (sum(ia.runs_in_innings))::smallint AS "R",
    (max(ia.runs_in_innings))::smallint AS "HS",
        CASE
            WHEN (sum(
            CASE
                WHEN (ia.is_out = 1) THEN 1
                ELSE 0
            END) = 0) THEN (0)::double precision
            ELSE (round((sum(ia.runs_in_innings) / (sum(
            CASE
                WHEN (ia.is_out = 1) THEN 1
                ELSE 0
            END))::numeric), 2))::double precision
        END AS "Avg",
        CASE
            WHEN (sum(ia.balls_faced_in_innings) = (0)::numeric) THEN (0)::double precision
            ELSE (round(((sum(ia.runs_in_innings) / sum(ia.balls_faced_in_innings)) * (100)::numeric), 2))::double precision
        END AS "SR",
    (sum(
        CASE
            WHEN (ia.runs_in_innings >= 100) THEN 1
            ELSE 0
        END))::smallint AS "100s",
    (sum(
        CASE
            WHEN ((ia.runs_in_innings >= 50) AND (ia.runs_in_innings <= 99)) THEN 1
            ELSE 0
        END))::smallint AS "50s",
    (sum(
        CASE
            WHEN ((ia.runs_in_innings = 0) AND (ia.is_out = 1)) THEN 1
            ELSE 0
        END))::smallint AS "Ducks",
    (sum(ia.balls_faced_in_innings))::smallint AS "BF",
    (sum(ia.boundary_hits))::smallint AS "Boundaries",
    (sum(
        CASE
            WHEN (ia.is_out = 0) THEN 1
            ELSE 0
        END))::smallint AS "NO"
   FROM innings_agg ia
  GROUP BY ia.player_id;


ALTER TABLE public.batter_stats OWNER TO postgres;

--
-- Name: bowler_stats; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.bowler_stats AS
 WITH ball_agg AS (
         SELECT b.bowler_id AS player_id,
            b.match_id,
            b.innings_num,
            b.over_num,
                CASE
                    WHEN ((e.extra_type)::text = ANY ((ARRAY['no_ball'::character varying, 'wide'::character varying])::text[])) THEN 1
                    ELSE 1
                END AS ball_delivered,
            COALESCE((bs.run_scored)::integer, 0) AS runs_batter,
            COALESCE((e.extra_runs)::integer, 0) AS runs_extra,
                CASE
                    WHEN ((w.kind_out)::text = ANY ((ARRAY['bowled'::character varying, 'caught'::character varying, 'lbw'::character varying, 'stumped'::character varying])::text[])) THEN 1
                    ELSE 0
                END AS is_wicket
           FROM (((public.balls b
             LEFT JOIN public.batter_score bs ON ((((b.match_id)::text = (bs.match_id)::text) AND (b.innings_num = bs.innings_num) AND (b.over_num = bs.over_num) AND (b.ball_num = bs.ball_num))))
             LEFT JOIN public.extras e ON ((((b.match_id)::text = (e.match_id)::text) AND (b.innings_num = e.innings_num) AND (b.over_num = e.over_num) AND (b.ball_num = e.ball_num))))
             LEFT JOIN public.wickets w ON ((((b.match_id)::text = (w.match_id)::text) AND (b.innings_num = w.innings_num) AND (b.over_num = w.over_num) AND (b.ball_num = w.ball_num))))
        )
 SELECT ball_agg.player_id,
    (sum(ball_agg.ball_delivered))::smallint AS "B",
    (sum(ball_agg.is_wicket))::smallint AS "W",
    (sum((ball_agg.runs_batter + ball_agg.runs_extra)))::smallint AS "Runs",
        CASE
            WHEN (sum(ball_agg.is_wicket) = 0) THEN (0)::double precision
            ELSE (round(((sum((ball_agg.runs_batter + ball_agg.runs_extra)))::numeric / (sum(ball_agg.is_wicket))::numeric), 2))::double precision
        END AS "Avg",
        CASE
            WHEN (count(DISTINCT ROW(ball_agg.match_id, ball_agg.innings_num, ball_agg.over_num)) = 0) THEN (0)::double precision
            ELSE (round(((sum((ball_agg.runs_batter + ball_agg.runs_extra)))::numeric / (count(DISTINCT ROW(ball_agg.match_id, ball_agg.innings_num, ball_agg.over_num)))::numeric), 2))::double precision
        END AS "Econ",
        CASE
            WHEN (sum(ball_agg.is_wicket) = 0) THEN (0)::double precision
            ELSE (round(((sum(ball_agg.ball_delivered))::numeric / (sum(ball_agg.is_wicket))::numeric), 2))::double precision
        END AS "SR",
    (sum(ball_agg.runs_extra))::smallint AS "Extras"
   FROM ball_agg
  GROUP BY ball_agg.player_id;


ALTER TABLE public.bowler_stats OWNER TO postgres;

--
-- Name: fielder_stats; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.fielder_stats AS
 SELECT w.fielder_id AS player_id,
    (sum(
        CASE
            WHEN ((w.kind_out)::text = 'caught'::text) THEN 1
            ELSE 0
        END))::smallint AS "C",
    (sum(
        CASE
            WHEN ((w.kind_out)::text = 'stumped'::text) THEN 1
            ELSE 0
        END))::smallint AS "St",
    (sum(
        CASE
            WHEN ((w.kind_out)::text = 'runout'::text) THEN 1
            ELSE 0
        END))::smallint AS "RO"
   FROM public.wickets w
  WHERE (w.fielder_id IS NOT NULL)
  GROUP BY w.fielder_id;


ALTER TABLE public.fielder_stats OWNER TO postgres;

--
-- Name: match; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.match (
    match_id character varying(20) NOT NULL,
    match_type character varying(20) NOT NULL,
    venue character varying(20) NOT NULL,
    team_1_id character varying(20) NOT NULL,
    team_2_id character varying(20) NOT NULL,
    match_date date NOT NULL,
    season_id character varying(20) NOT NULL,
    win_run_margin smallint,
    win_by_wickets smallint,
    win_type character varying(20),
    toss_winner smallint,
    toss_decide character varying(20),
    winner_team_id character varying(20),
    CONSTRAINT match_check CHECK (((win_type IS NULL) OR (((win_type)::text = 'draw'::text) AND (win_run_margin IS NULL) AND (win_by_wickets IS NULL)) OR (((win_type)::text = 'runs'::text) AND (win_run_margin IS NOT NULL) AND (win_by_wickets IS NULL)) OR (((win_type)::text = 'wickets'::text) AND (win_run_margin IS NULL) AND (win_by_wickets IS NOT NULL)))),
    CONSTRAINT match_match_type_check CHECK (((match_type)::text = ANY ((ARRAY['league'::character varying, 'playoff'::character varying, 'knockout'::character varying])::text[]))),
    CONSTRAINT match_toss_decide_check CHECK (((toss_decide)::text = ANY ((ARRAY['bowl'::character varying, 'bat'::character varying])::text[]))),
    CONSTRAINT match_toss_winner_check CHECK ((toss_winner = ANY (ARRAY[1, 2]))),
    CONSTRAINT match_win_type_check CHECK (((win_type)::text = ANY ((ARRAY['runs'::character varying, 'wickets'::character varying, 'draw'::character varying])::text[])))
);


ALTER TABLE public.match OWNER TO postgres;

--
-- Name: player; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.player (
    player_id character varying(20) NOT NULL,
    player_name character varying(255) NOT NULL,
    dob date NOT NULL,
    batting_hand character varying(20) NOT NULL,
    bowling_skill character varying(20),
    country_name character varying(20) NOT NULL,
    CONSTRAINT player_batting_hand_check CHECK (((batting_hand)::text = ANY ((ARRAY['left'::character varying, 'right'::character varying])::text[]))),
    CONSTRAINT player_bowling_skill_check CHECK (((bowling_skill)::text = ANY ((ARRAY['fast'::character varying, 'medium'::character varying, 'legspin'::character varying, 'offspin'::character varying])::text[]))),
    CONSTRAINT player_dob_check CHECK ((dob < '2016-01-01'::date))
);


ALTER TABLE public.player OWNER TO postgres;

--
-- Name: player_match; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.player_match (
    player_id character varying(20) NOT NULL,
    match_id character varying(20) NOT NULL,
    role character varying(20) NOT NULL,
    team_id character varying(20) NOT NULL,
    is_extra boolean NOT NULL,
    CONSTRAINT player_match_role_check CHECK (((role)::text = ANY ((ARRAY['batter'::character varying, 'bowler'::character varying, 'allrounder'::character varying, 'wicketkeeper'::character varying])::text[])))
);


ALTER TABLE public.player_match OWNER TO postgres;

--
-- Name: player_team; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.player_team (
    player_id character varying(20) NOT NULL,
    team_id character varying(20) NOT NULL,
    season_id character varying(20) NOT NULL
);


ALTER TABLE public.player_team OWNER TO postgres;

--
-- Name: season; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.season (
    season_id character varying(20) NOT NULL,
    year smallint NOT NULL,
    start_date date NOT NULL,
    end_date date NOT NULL,
    CONSTRAINT season_year_check CHECK (((year >= 1900) AND (year <= 2025)))
);


ALTER TABLE public.season OWNER TO postgres;

--
-- Name: team; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.team (
    team_id character varying(20) NOT NULL,
    team_name character varying(255) NOT NULL,
    coach_name character varying(255) NOT NULL,
    region character varying(20) NOT NULL
);


ALTER TABLE public.team OWNER TO postgres;

--
-- Data for Name: auction; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) FROM stdin;
\.


--
-- Data for Name: awards; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.awards (match_id, award_type, player_id) FROM stdin;
\.


--
-- Data for Name: balls; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) FROM stdin;
\.


--
-- Data for Name: batter_score; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.batter_score (match_id, over_num, innings_num, ball_num, run_scored, type_run) FROM stdin;
\.


--
-- Data for Name: extras; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.extras (match_id, innings_num, over_num, ball_num, extra_runs, extra_type) FROM stdin;
\.


--
-- Data for Name: match; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_run_margin, win_by_wickets, win_type, toss_winner, toss_decide, winner_team_id) FROM stdin;
\.


--
-- Data for Name: player; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.player (player_id, player_name, dob, batting_hand, bowling_skill, country_name) FROM stdin;
\.


--
-- Data for Name: player_match; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.player_match (player_id, match_id, role, team_id, is_extra) FROM stdin;
\.


--
-- Data for Name: player_team; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.player_team (player_id, team_id, season_id) FROM stdin;
\.


--
-- Data for Name: season; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.season (season_id, year, start_date, end_date) FROM stdin;
IPL2025	2025	2025-01-01	2025-12-31
\.


--
-- Data for Name: team; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.team (team_id, team_name, coach_name, region) FROM stdin;
\.


--
-- Data for Name: wickets; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out, fielder_id) FROM stdin;
\.


--
-- Name: auction auction_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auction
    ADD CONSTRAINT auction_pkey PRIMARY KEY (auction_id);


--
-- Name: auction auction_player_id_team_id_season_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auction
    ADD CONSTRAINT auction_player_id_team_id_season_id_key UNIQUE (player_id, team_id, season_id);


--
-- Name: awards awards_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.awards
    ADD CONSTRAINT awards_pkey PRIMARY KEY (match_id, award_type);


--
-- Name: balls balls_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.balls
    ADD CONSTRAINT balls_pkey PRIMARY KEY (match_id, innings_num, over_num, ball_num);


--
-- Name: batter_score batter_score_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.batter_score
    ADD CONSTRAINT batter_score_pkey PRIMARY KEY (match_id, innings_num, over_num, ball_num);


--
-- Name: extras extras_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.extras
    ADD CONSTRAINT extras_pkey PRIMARY KEY (match_id, innings_num, over_num, ball_num);


--
-- Name: match match_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.match
    ADD CONSTRAINT match_pkey PRIMARY KEY (match_id);


--
-- Name: player_match player_match_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.player_match
    ADD CONSTRAINT player_match_pkey PRIMARY KEY (player_id, match_id);


--
-- Name: player player_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.player
    ADD CONSTRAINT player_pkey PRIMARY KEY (player_id);


--
-- Name: player_team player_team_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.player_team
    ADD CONSTRAINT player_team_pkey PRIMARY KEY (player_id, team_id, season_id);


--
-- Name: season season_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.season
    ADD CONSTRAINT season_pkey PRIMARY KEY (season_id);


--
-- Name: team team_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.team
    ADD CONSTRAINT team_pkey PRIMARY KEY (team_id);


--
-- Name: team team_region_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.team
    ADD CONSTRAINT team_region_key UNIQUE (region);


--
-- Name: team team_team_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.team
    ADD CONSTRAINT team_team_name_key UNIQUE (team_name);


--
-- Name: wickets wickets_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.wickets
    ADD CONSTRAINT wickets_pkey PRIMARY KEY (match_id, innings_num, over_num, ball_num);


--
-- Name: auction auction_deletion_trigger; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER auction_deletion_trigger AFTER DELETE ON public.auction FOR EACH ROW EXECUTE FUNCTION public.auction_deletion_cleanup();


--
-- Name: match automated_match_id_validation; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER automated_match_id_validation BEFORE INSERT OR UPDATE ON public.match FOR EACH ROW EXECUTE FUNCTION public.match_id_validation();


--
-- Name: auction automated_player_team_insertion; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER automated_player_team_insertion BEFORE INSERT OR UPDATE ON public.auction FOR EACH ROW EXECUTE FUNCTION public.automatic_insertion_into_player_team();


--
-- Name: season automated_season_id_generation; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER automated_season_id_generation BEFORE INSERT ON public.season FOR EACH ROW EXECUTE FUNCTION public.automatic_season_id_generation();


--
-- Name: wickets enforce_wicketkeeper_for_stumped; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER enforce_wicketkeeper_for_stumped BEFORE INSERT OR UPDATE ON public.wickets FOR EACH ROW EXECUTE FUNCTION public.check_wicketkeeper_for_stumped();


--
-- Name: match home_match_count_constraint; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER home_match_count_constraint BEFORE INSERT OR UPDATE ON public.match FOR EACH ROW EXECUTE FUNCTION public.limit_on_number_of_home_matches();


--
-- Name: match insert_awards_trigger; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER insert_awards_trigger AFTER UPDATE ON public.match FOR EACH ROW EXECUTE FUNCTION public.insert_awards_after_match();


--
-- Name: player_team international_player_count_constraint; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER international_player_count_constraint BEFORE INSERT OR UPDATE ON public.player_team FOR EACH ROW EXECUTE FUNCTION public.limit_on_international_players_per_team();


--
-- Name: match match_deletion_trigger; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER match_deletion_trigger AFTER DELETE ON public.match FOR EACH ROW EXECUTE FUNCTION public.match_deletion_cleanup();


--
-- Name: season season_deletion_trigger; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER season_deletion_trigger AFTER DELETE ON public.season FOR EACH ROW EXECUTE FUNCTION public.season_deletion_cleanup();


--
-- Name: match set_winner_team_trigger; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER set_winner_team_trigger BEFORE UPDATE ON public.match FOR EACH ROW EXECUTE FUNCTION public.set_winner_team_id();


--
-- Name: auction auction_player_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auction
    ADD CONSTRAINT auction_player_id_fkey FOREIGN KEY (player_id) REFERENCES public.player(player_id);


--
-- Name: auction auction_season_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auction
    ADD CONSTRAINT auction_season_id_fkey FOREIGN KEY (season_id) REFERENCES public.season(season_id);


--
-- Name: auction auction_team_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auction
    ADD CONSTRAINT auction_team_id_fkey FOREIGN KEY (team_id) REFERENCES public.team(team_id);


--
-- Name: awards awards_match_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.awards
    ADD CONSTRAINT awards_match_id_fkey FOREIGN KEY (match_id) REFERENCES public.match(match_id);


--
-- Name: awards awards_player_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.awards
    ADD CONSTRAINT awards_player_id_fkey FOREIGN KEY (player_id) REFERENCES public.player(player_id);


--
-- Name: balls balls_bowler_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.balls
    ADD CONSTRAINT balls_bowler_id_fkey FOREIGN KEY (bowler_id) REFERENCES public.player(player_id);


--
-- Name: balls balls_match_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.balls
    ADD CONSTRAINT balls_match_id_fkey FOREIGN KEY (match_id) REFERENCES public.match(match_id);


--
-- Name: balls balls_non_striker_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.balls
    ADD CONSTRAINT balls_non_striker_id_fkey FOREIGN KEY (non_striker_id) REFERENCES public.player(player_id);


--
-- Name: balls balls_striker_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.balls
    ADD CONSTRAINT balls_striker_id_fkey FOREIGN KEY (striker_id) REFERENCES public.player(player_id);


--
-- Name: batter_score batter_score_match_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.batter_score
    ADD CONSTRAINT batter_score_match_id_fkey FOREIGN KEY (match_id) REFERENCES public.match(match_id);


--
-- Name: batter_score batter_score_match_id_innings_num_over_num_ball_num_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.batter_score
    ADD CONSTRAINT batter_score_match_id_innings_num_over_num_ball_num_fkey FOREIGN KEY (match_id, innings_num, over_num, ball_num) REFERENCES public.balls(match_id, innings_num, over_num, ball_num);


--
-- Name: extras extras_match_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.extras
    ADD CONSTRAINT extras_match_id_fkey FOREIGN KEY (match_id) REFERENCES public.match(match_id);


--
-- Name: extras extras_match_id_innings_num_over_num_ball_num_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.extras
    ADD CONSTRAINT extras_match_id_innings_num_over_num_ball_num_fkey FOREIGN KEY (match_id, innings_num, over_num, ball_num) REFERENCES public.balls(match_id, innings_num, over_num, ball_num);


--
-- Name: match match_season_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.match
    ADD CONSTRAINT match_season_id_fkey FOREIGN KEY (season_id) REFERENCES public.season(season_id);


--
-- Name: match match_team_1_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.match
    ADD CONSTRAINT match_team_1_id_fkey FOREIGN KEY (team_1_id) REFERENCES public.team(team_id);


--
-- Name: match match_team_2_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.match
    ADD CONSTRAINT match_team_2_id_fkey FOREIGN KEY (team_2_id) REFERENCES public.team(team_id);


--
-- Name: match match_venue_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.match
    ADD CONSTRAINT match_venue_fkey FOREIGN KEY (venue) REFERENCES public.team(region);


--
-- Name: match match_winner_team_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.match
    ADD CONSTRAINT match_winner_team_id_fkey FOREIGN KEY (winner_team_id) REFERENCES public.team(team_id);


--
-- Name: player_match player_match_match_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.player_match
    ADD CONSTRAINT player_match_match_id_fkey FOREIGN KEY (match_id) REFERENCES public.match(match_id);


--
-- Name: player_match player_match_player_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.player_match
    ADD CONSTRAINT player_match_player_id_fkey FOREIGN KEY (player_id) REFERENCES public.player(player_id);


--
-- Name: player_match player_match_team_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.player_match
    ADD CONSTRAINT player_match_team_id_fkey FOREIGN KEY (team_id) REFERENCES public.team(team_id);


--
-- Name: player_team player_team_player_id_team_id_season_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.player_team
    ADD CONSTRAINT player_team_player_id_team_id_season_id_fkey FOREIGN KEY (player_id, team_id, season_id) REFERENCES public.auction(player_id, team_id, season_id);


--
-- Name: wickets wickets_fielder_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.wickets
    ADD CONSTRAINT wickets_fielder_id_fkey FOREIGN KEY (fielder_id) REFERENCES public.player(player_id);


--
-- Name: wickets wickets_match_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.wickets
    ADD CONSTRAINT wickets_match_id_fkey FOREIGN KEY (match_id) REFERENCES public.match(match_id);


--
-- Name: wickets wickets_match_id_innings_num_over_num_ball_num_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.wickets
    ADD CONSTRAINT wickets_match_id_innings_num_over_num_ball_num_fkey FOREIGN KEY (match_id, innings_num, over_num, ball_num) REFERENCES public.balls(match_id, innings_num, over_num, ball_num);


--
-- Name: wickets wickets_player_out_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.wickets
    ADD CONSTRAINT wickets_player_out_id_fkey FOREIGN KEY (player_out_id) REFERENCES public.player(player_id);


--
-- PostgreSQL database dump complete
--

