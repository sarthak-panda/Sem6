SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = ON;
SET search_path = public;
SET check_function_bodies = FALSE;
SET xmloption = CONTENT;
SET client_min_messages = warning;
SET row_security = OFF;

CREATE SCHEMA PUBLIC;

SET default_tablespace = '';
SET default_table_access_method = heap;

CREATE TABLE public.player (
    player_id VARCHAR(20) NOT NULL,
    player_name VARCHAR(255) NOT NULL,
    dob DATE NOT NULL CHECK (dob < '2016-01-01'),
    batting_hand VARCHAR(20) NOT NULL CHECK (batting_hand IN ('left', 'right')),
    bowling_skill VARCHAR(20) CHECK (bowling_skill IN ('fast', 'medium', 'legspin', 'offspin')),
    country_name VARCHAR(20) NOT NULL,
    PRIMARY KEY (player_id)
);

CREATE TABLE public.season (
    season_id VARCHAR(20) NOT NULL,
    year SMALLINT NOT NULL CHECK (year BETWEEN 1900 AND 2025),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    PRIMARY KEY (season_id)
);

CREATE TABLE public.team (
    team_id VARCHAR(20) NOT NULL,
    team_name VARCHAR(255) NOT NULL UNIQUE,
    coach_name VARCHAR(255) NOT NULL,
    region VARCHAR(20) NOT NULL UNIQUE,
    PRIMARY KEY (team_id)
);

CREATE TABLE public.auction (
	auction_id VARCHAR(20) NOT NULL,
	season_id VARCHAR(20) NOT NULL,
	player_id VARCHAR(20) NOT NULL,
	base_price BIGINT NOT NULL CHECK (base_price >= 1000000),
	sold_price BIGINT,
	is_sold BOOLEAN NOT NULL,
	team_id VARCHAR(20),
	PRIMARY KEY (auction_id),
	FOREIGN KEY (season_id) REFERENCES public.season (season_id),
	FOREIGN KEY (player_id) REFERENCES public.player (player_id),
	FOREIGN KEY (team_id) REFERENCES public.team (team_id),
    UNIQUE (player_id, team_id, season_id),
    CHECK (
        (is_sold = FALSE AND sold_price IS NULL)--O
        OR
        (is_sold = TRUE AND sold_price IS NOT NULL AND team_id IS NOT NULL AND sold_price >= base_price)
    )
);

CREATE TABLE public.match (
    match_id VARCHAR(20) NOT NULL,
    match_type VARCHAR(20) NOT NULL CHECK (match_type IN ('league', 'playoff', 'knockout')),
    venue VARCHAR(20) NOT NULL,
    team_1_id VARCHAR(20) NOT NULL,
    team_2_id VARCHAR(20) NOT NULL,
    match_date DATE NOT NULL,
    season_id VARCHAR(20) NOT NULL,
    win_run_margin SMALLINT,
    win_by_wickets SMALLINT,
    win_type VARCHAR(20) CHECK (win_type IN ('runs', 'wickets', 'draw')),
    toss_winner SMALLINT CHECK (toss_winner IN (1, 2)),
    toss_decide VARCHAR(20) CHECK (toss_decide IN ('bowl', 'bat')),
    winner_team_id VARCHAR(20),
    PRIMARY KEY (match_id),
    FOREIGN KEY (venue) REFERENCES public.team(region),
    FOREIGN KEY (team_1_id) REFERENCES public.team(team_id),
    FOREIGN KEY (team_2_id) REFERENCES public.team(team_id),
    FOREIGN KEY (season_id) REFERENCES public.season(season_id),
    FOREIGN KEY (winner_team_id) REFERENCES public.team(team_id),
    -- CHECK Constraints for win_type and margins
    CHECK (
        (win_type IS NULL)
        OR
        (win_type = 'draw' AND win_run_margin IS NULL AND win_by_wickets IS NULL)
        OR
        (win_type = 'runs' AND win_run_margin IS NOT NULL AND win_by_wickets IS NULL)
        OR
        (win_type = 'wickets' AND win_run_margin IS NULL AND win_by_wickets IS NOT NULL)
    )    
);

CREATE TABLE public.awards (
	match_id VARCHAR(20) NOT NULL,
	award_type VARCHAR(20) NOT NULL CHECK (award_type IN ('orange_cap', 'purple_cap')),
	player_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (match_id, award_type),
    FOREIGN KEY (match_id) REFERENCES public.match (match_id),
    FOREIGN KEY (player_id) REFERENCES public.player (player_id)	
);

CREATE TABLE public.player_team (
    player_id VARCHAR(20) NOT NULL,
    team_id VARCHAR(20) NOT NULL,
    season_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (player_id, team_id, season_id),
    FOREIGN KEY (player_id, team_id, season_id) REFERENCES public.auction(player_id, team_id, season_id)
);

CREATE TABLE public.balls (
    match_id VARCHAR(20) NOT NULL,
    innings_num SMALLINT NOT NULL,
    over_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    striker_id VARCHAR(20) NOT NULL,
    non_striker_id VARCHAR(20) NOT NULL,
    bowler_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (match_id, innings_num, over_num, ball_num),
    FOREIGN KEY (match_id) REFERENCES public.match(match_id),
    FOREIGN KEY (striker_id) REFERENCES public.player(player_id),
    FOREIGN KEY (non_striker_id) REFERENCES public.player(player_id),
    FOREIGN KEY (bowler_id) REFERENCES public.player(player_id)
);

CREATE TABLE public.player_match (
    player_id VARCHAR(20) NOT NULL,
    match_id VARCHAR(20) NOT NULL,
    ROLE VARCHAR(20) NOT NULL CHECK (ROLE IN ('batter', 'bowler', 'allrounder', 'wicketkeeper')),
    team_id VARCHAR(20) NOT NULL,
    is_extra BOOLEAN NOT NULL,
    PRIMARY KEY (player_id, match_id),
    FOREIGN KEY (player_id) REFERENCES public.player(player_id),
    FOREIGN KEY (match_id) REFERENCES public.match(match_id),
    FOREIGN KEY (team_id) REFERENCES public.team(team_id)
);

CREATE TABLE public.wickets (
    match_id VARCHAR(20) NOT NULL,
    innings_num SMALLINT NOT NULL,
    over_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    player_out_id VARCHAR(20) NOT NULL,
    kind_out VARCHAR(20) NOT NULL CHECK (kind_out IN ('bowled', 'caught', 'lbw', 'runout', 'stumped', 'hitwicket')),
    fielder_id VARCHAR(20),
    PRIMARY KEY (match_id, innings_num, over_num, ball_num),
    FOREIGN KEY (match_id) REFERENCES public.match(match_id),
    FOREIGN KEY (player_out_id) REFERENCES public.player(player_id),
    FOREIGN KEY (fielder_id) REFERENCES public.player(player_id),
    FOREIGN KEY (match_id, innings_num, over_num, ball_num) REFERENCES public.balls(match_id, innings_num, over_num, ball_num), -- Composite FK
    CHECK (
        (kind_out IN ('caught', 'runout', 'stumped') AND fielder_id IS NOT NULL)
        OR
        (kind_out NOT IN ('caught', 'runout', 'stumped') AND fielder_id IS NULL)
    )    
);

CREATE TABLE public.extras (
    match_id VARCHAR(20) NOT NULL,
    innings_num SMALLINT NOT NULL,
    over_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    extra_runs SMALLINT NOT NULL CHECK (extra_runs >= 0),
    extra_type VARCHAR(20) NOT NULL CHECK (extra_type IN ('no_ball', 'wide', 'byes', 'legbyes')),
    PRIMARY KEY (match_id, innings_num, over_num, ball_num),
    FOREIGN KEY (match_id) REFERENCES public.match(match_id),
    FOREIGN KEY (match_id, innings_num, over_num, ball_num) REFERENCES public.balls(match_id, innings_num, over_num, ball_num) -- Composite FK
);

CREATE TABLE public.batter_score (
    match_id VARCHAR(20) NOT NULL,
    over_num SMALLINT NOT NULL,
    innings_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    run_scored SMALLINT NOT NULL CHECK (run_scored >= 0),
    type_run VARCHAR(20) CHECK (type_run IN ('running', 'boundary')),
    PRIMARY KEY (match_id, innings_num, over_num, ball_num),
    FOREIGN KEY (match_id) REFERENCES public.match(match_id),
    FOREIGN KEY (match_id, innings_num, over_num, ball_num) REFERENCES public.balls(match_id, innings_num, over_num, ball_num) -- Composite FK
);

CREATE OR REPLACE FUNCTION automatic_insertion_into_player_team()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_sold = TRUE THEN
        --i am assuming we do not need to update because only insert after auction done policy
        INSERT INTO public.player_team(player_id,team_id,season_id) VALUES (NEW.player_id,NEW.team_id,NEW.season_id);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER automated_player_team_insertion
BEFORE INSERT OR UPDATE ON public.auction --O
FOR EACH ROW
EXECUTE FUNCTION automatic_insertion_into_player_team();

CREATE OR REPLACE FUNCTION match_id_validation()
RETURNS TRIGGER AS $$
DECLARE
    season_part VARCHAR(20);
    seq_part INTEGER;
    prev_match_id VARCHAR(20);
BEGIN
    -- IF NEW.match_id IS NULL THEN
    --     RAISE EXCEPTION 'null';--check--changed
    -- END IF;
    IF NEW.match_id !~ '^[a-zA-Z0-9]+[0-9]{3}$' THEN
        RAISE EXCEPTION 'sequence of match id violated null';
    END IF;
    season_part := LEFT(NEW.match_id, LENGTH(NEW.match_id) - 3);
    seq_part := CAST(RIGHT(NEW.match_id, 3) AS INTEGER);
    IF NEW.season_id <> season_part THEN
        RAISE EXCEPTION 'sequence of match id violated null';
    END IF;
    IF seq_part = 1 THEN
        IF EXISTS (SELECT 1 FROM public.match WHERE match_id = NEW.match_id) THEN
            RAISE EXCEPTION 'sequence of match id violated null';
        END IF;
    ELSE
        prev_match_id := season_part || LPAD((seq_part - 1)::TEXT, 3, '0');
        IF NOT EXISTS (SELECT 1 FROM public.match WHERE match_id = prev_match_id) THEN
            RAISE EXCEPTION 'sequence of match id violated null';
        END IF;
        IF EXISTS (SELECT 1 FROM public.match WHERE match_id = NEW.match_id) THEN
            RAISE EXCEPTION 'sequence of match id violated null';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER automated_match_id_validation
BEFORE INSERT OR UPDATE ON public.match
FOR EACH ROW
EXECUTE FUNCTION match_id_validation();

CREATE OR REPLACE FUNCTION automatic_season_id_generation()
RETURNS TRIGGER AS $$
BEGIN
    --NEW.season_id := 'IPL' || NEW.year;
    NEW.season_id := CONCAT('IPL', NEW.year)::varchar(20);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER automated_season_id_generation
BEFORE INSERT ON public.season
FOR EACH ROW
EXECUTE FUNCTION automatic_season_id_generation();

CREATE OR REPLACE FUNCTION check_wicketkeeper_for_stumped()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.kind_out = 'stumped' THEN
        IF NOT EXISTS (
            SELECT 1 FROM public.player_match pm
            WHERE pm.match_id = NEW.match_id
            AND pm.player_id = NEW.fielder_id
            AND pm.role = 'wicketkeeper'
        ) THEN
            RAISE EXCEPTION 'for stumped dismissal, fielder must be a wicketkeeper null';
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_wicketkeeper_for_stumped
BEFORE INSERT OR UPDATE ON public.wickets
FOR EACH ROW
EXECUTE FUNCTION check_wicketkeeper_for_stumped();

CREATE OR REPLACE FUNCTION limit_on_international_players_per_team()
RETURNS TRIGGER AS $$
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
            RAISE EXCEPTION 'there could be atmost 3 international players per team per season null';
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER international_player_count_constraint
BEFORE INSERT OR UPDATE ON public.player_team
FOR EACH ROW
EXECUTE FUNCTION limit_on_international_players_per_team();

CREATE OR REPLACE FUNCTION limit_on_number_of_home_matches()
RETURNS TRIGGER AS $$
DECLARE
    home_region_team1 VARCHAR(20);
    home_region_team2 VARCHAR(20);
BEGIN
    SELECT region INTO home_region_team1 FROM public.team WHERE team_id = NEW.team_1_id;
    SELECT region INTO home_region_team2 FROM public.team WHERE team_id = NEW.team_2_id;
    -- League match must be played at home ground of one of the teams
    IF NEW.match_type = 'league' THEN
        IF NEW.venue <> home_region_team1 AND NEW.venue <> home_region_team2 THEN
            RAISE EXCEPTION 'league match must be played at home ground of one of the teams null';
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
                RAISE EXCEPTION 'each team can play only one home match in a league against another team null';
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
                RAISE EXCEPTION 'each team can play only one home match in a league against another team null';
            END IF;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach Trigger to match table
CREATE TRIGGER home_match_count_constraint
BEFORE INSERT OR UPDATE ON public.match
FOR EACH ROW
EXECUTE FUNCTION limit_on_number_of_home_matches();

--UPDATING OF ROWS
CREATE OR REPLACE FUNCTION set_winner_team_id()
RETURNS TRIGGER AS $$
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
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_winner_team_trigger
BEFORE UPDATE ON public.match
FOR EACH ROW
EXECUTE FUNCTION set_winner_team_id();

CREATE OR REPLACE FUNCTION insert_awards_after_match()
RETURNS TRIGGER AS $$
DECLARE
    orange_cap_player VARCHAR(20);
    purple_cap_player VARCHAR(20);
BEGIN
    IF NEW.win_type IS NOT NULL AND (OLD.win_type IS DISTINCT FROM NEW.win_type) THEN

        WITH runs_o AS (
            SELECT striker_id, SUM(run_scored) AS total_runs
            FROM public.balls b 
            JOIN public.batter_score bs 
                ON b.match_id = bs.match_id AND b.innings_num = bs.innings_num AND b.over_num = bs.over_num AND b.ball_num = bs.ball_num
            WHERE match_id = NEW.match_id
            GROUP BY striker_id
        ),
        ranked_runs AS (
            SELECT striker_id
            FROM runs_o
            ORDER BY total_runs DESC, striker_id ASC
            LIMIT 1
        )
        SELECT striker_id INTO orange_cap_player
        FROM ranked_runs;

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
            LIMIT 1
        )
        SELECT bowler_id INTO purple_cap_player
        FROM ranked_wickets;
        
        IF orange_cap_player IS NOT NULL THEN
            INSERT INTO public.awards (match_id, award_type, player_id) VALUES (NEW.match_id, 'orange_cap', orange_cap_player);
        END IF;
        IF purple_cap_player IS NOT NULL THEN
            INSERT INTO public.awards (match_id, award_type, player_id) VALUES (NEW.match_id, 'purple_cap', purple_cap_player);
        END IF;
        -- INSERT INTO public.awards (match_id, award_type, player_id)
        -- VALUES
        --     (NEW.match_id, 'orange_cap', orange_cap_player),
        --     (NEW.match_id, 'purple_cap', purple_cap_player);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER insert_awards_trigger
AFTER UPDATE ON public.match
FOR EACH ROW
EXECUTE FUNCTION insert_awards_after_match();

--DELETION OF ROWS
--AUCTION DELETION
CREATE OR REPLACE FUNCTION auction_deletion_cleanup()
RETURNS TRIGGER AS $$
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

        DELETE FROM public.wickets w
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
$$ LANGUAGE plpgsql;

CREATE TRIGGER auction_deletion_trigger
BEFORE DELETE ON public.auction
FOR EACH ROW
EXECUTE FUNCTION auction_deletion_cleanup();

--MATCH DELETION
CREATE OR REPLACE FUNCTION match_deletion_cleanup()
RETURNS TRIGGER AS $$
BEGIN
    DELETE FROM public.awards
    WHERE match_id = OLD.match_id;

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
$$ LANGUAGE plpgsql;

CREATE TRIGGER match_deletion_trigger
BEFORE DELETE ON public.match
FOR EACH ROW
EXECUTE FUNCTION match_deletion_cleanup();

--SEASON DELETION
CREATE OR REPLACE FUNCTION season_deletion_cleanup()
RETURNS TRIGGER AS $$
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
$$ LANGUAGE plpgsql;

CREATE TRIGGER season_deletion_trigger
BEFORE DELETE ON public.season
FOR EACH ROW
EXECUTE FUNCTION season_deletion_cleanup();

--TO CREATE VIEWS
--batter stats
CREATE OR REPLACE VIEW public.batter_stats AS
WITH innings_agg AS (
    SELECT
        b.striker_id AS player_id, b.match_id, b.innings_num,
        -- ignoring extras in run count
        COALESCE(SUM(bs.run_scored), 0) AS runs_in_innings,
        -- balls faced if they do NOT appear in extras
        COUNT(*) FILTER (WHERE e.match_id IS NULL) AS balls_faced_in_innings,
        SUM(
            CASE 
                WHEN bs.type_run = 'boundary' AND bs.run_scored IN (4, 6) 
                THEN 1 
                ELSE 0 
            END
        ) AS boundary_hits,
        CASE 
            WHEN MAX(w.player_out_id) = b.striker_id THEN 1 
            ELSE 0 
        END AS is_out     
    FROM public.balls b
    LEFT JOIN public.batter_score bs
            ON b.match_id = bs.match_id AND b.innings_num = bs.innings_num AND b.over_num = bs.over_num AND b.ball_num = bs.ball_num
    LEFT JOIN public.extras e
            ON b.match_id = e.match_id AND b.innings_num = e.innings_num AND b.over_num = e.over_num AND b.ball_num = e.ball_num
    LEFT JOIN public.wickets w
            ON b.match_id = w.match_id AND b.innings_num = w.innings_num AND b.over_num = w.over_num AND b.ball_num = w.ball_num
            AND w.player_out_id = b.striker_id
    GROUP BY b.striker_id, b.match_id, b.innings_num
)
SELECT
    ia.player_id::varchar(20) AS player_id,
    COUNT(DISTINCT ia.match_id)::smallint AS "Mat",
    COUNT(*)::smallint AS "Inns",
    SUM(ia.runs_in_innings)::smallint AS "R",
    MAX(ia.runs_in_innings)::smallint AS "HS",
    CASE 
        WHEN SUM(CASE WHEN ia.is_out = 1 THEN 1 ELSE 0 END) = 0 
            THEN 0::double precision 
            ELSE ROUND(SUM(ia.runs_in_innings)::numeric/SUM(CASE WHEN ia.is_out = 1 THEN 1 ELSE 0 END),2)::double precision
    END AS "Avg",
    CASE 
        WHEN SUM(ia.balls_faced_in_innings) = 0 
            THEN 0::double precision 
            ELSE ROUND((SUM(ia.runs_in_innings)::numeric/SUM(ia.balls_faced_in_innings))*100,2)::double precision
    END AS "SR",
    SUM(CASE WHEN ia.runs_in_innings >=100 THEN 1 ELSE 0 END)::smallint AS "100s",
    SUM(CASE WHEN ia.runs_in_innings BETWEEN 50 AND 99 THEN 1 ELSE 0 END)::smallint AS "50s",
    SUM(CASE WHEN ia.runs_in_innings = 0 AND ia.is_out = 1 THEN 1 ELSE 0 END)::smallint AS "Ducks",
    SUM(ia.balls_faced_in_innings)::smallint AS "BF",
    SUM(ia.boundary_hits)::smallint AS "Boundaries",
    SUM(CASE WHEN ia.is_out = 0 THEN 1 ELSE 0 END)::smallint AS "NO"
FROM innings_agg ia
GROUP BY ia.player_id;

--bowler stats
CREATE OR REPLACE VIEW public.bowler_stats AS
WITH ball_agg AS (
    SELECT
        b.bowler_id AS player_id,b.match_id,b.innings_num,b.over_num,
        CASE 
            WHEN e.extra_type IN ('no_ball', 'wide') THEN 1---0 
            ELSE 1 
        END AS ball_delivered,-- total deliveries excluding no-balls & wides
        COALESCE(bs.run_scored, 0) AS runs_batter,-- runs scored by the batter on this delivery
        COALESCE(e.extra_runs, 0) AS runs_extra,
        CASE -- total wickets of the type 'bowled','caught','lbw','stumped'
            WHEN w.kind_out IN ('bowled','caught','lbw','stumped') THEN 1
            ELSE 0 
        END AS is_wicket
    FROM public.balls b
    LEFT JOIN public.batter_score bs
            ON b.match_id = bs.match_id AND b.innings_num = bs.innings_num AND b.over_num = bs.over_num AND b.ball_num = bs.ball_num
    LEFT JOIN public.extras e
            ON b.match_id = e.match_id AND b.innings_num = e.innings_num AND b.over_num = e.over_num AND b.ball_num = e.ball_num
    LEFT JOIN public.wickets w
            ON b.match_id = w.match_id AND b.innings_num = w.innings_num AND b.over_num = w.over_num AND b.ball_num = w.ball_num
)
SELECT
    player_id::varchar(20) AS player_id,
    -- total deliveries excluding no-balls & wides
    SUM(ball_delivered)::smallint AS "B",
    -- total wickets of the type 'bowled','caught','lbw','stumped'
    SUM(is_wicket)::smallint AS "W",
    SUM(runs_batter + runs_extra)::smallint AS "Runs",
    CASE
        WHEN SUM(is_wicket) = 0 THEN 0::double precision
        ELSE ROUND(SUM(runs_batter + runs_extra)::numeric/SUM(is_wicket),2)::double precision
    END AS "Avg",
    CASE 
        WHEN COUNT(DISTINCT (match_id, innings_num, over_num)) = 0 THEN 0::double precision
        ELSE ROUND((SUM(runs_batter + runs_extra)::numeric/COUNT(DISTINCT (match_id, innings_num, over_num))),2)::double precision
    END AS "Econ",
    CASE 
        WHEN SUM(is_wicket) = 0 THEN 0::double precision
        ELSE ROUND((SUM(ball_delivered)::numeric/SUM(is_wicket)),2)::double precision
    END AS "SR",
    SUM(runs_extra)::smallint AS "Extras"
FROM ball_agg
GROUP BY player_id;

--fielder stats--types of out ('bowled', 'caught', 'lbw', 'runout', 'stumped', 'hitwicket'), i have a doubt hitwicket is neither used in bowler stat or fielder stat isn't it left out
CREATE OR REPLACE VIEW public.fielder_stats AS
SELECT
    w.fielder_id::varchar(20) AS player_id,
    SUM(CASE WHEN w.kind_out = 'caught'   THEN 1 ELSE 0 END)::smallint AS "C",
    -- Count how many times kind_out = 'stumped'
    SUM(CASE WHEN w.kind_out = 'stumped'  THEN 1 ELSE 0 END)::smallint AS "St",
    SUM(CASE WHEN w.kind_out = 'runout'   THEN 1 ELSE 0 END)::smallint AS "RO"
FROM public.wickets w
WHERE w.fielder_id IS NOT NULL
GROUP BY w.fielder_id;