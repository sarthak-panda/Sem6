SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = ON;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = FALSE;
SET xmloption = CONTENT;
SET client_min_messages = warning;
SET row_security = OFF;

CREATE SCHEMA PUBLIC;

SET default_tablespace = '';
SET default_table_access_method = heap;

CREATE TABLE public.auction (
	auction_id VARCHAR(20) NOT NULL,
	season_id VARCHAR(20) NOT NULL,
	player_id VARCHAR(20) NOT NULL,
	base_price BIGINT,
	sold_price BIGINT,
	is_sold BOOLEAN,
	team_id VARCHAR(20) NOT NULL,
	PRIMARY KEY (auction_id),
	FOREIGN KEY (season_id) REFERENCES public.season (season_id),
	FOREIGN KEY (player_id) REFERENCES public.player (player_id),
	FOREIGN KEY (team_id) REFERENCES public.team (team_id)
);

CREATE TABLE public.awards (
	match_id VARCHAR(20) NOT NULL,
	award_type VARCHAR(20) NOT NULL,
	player_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (match_id, award_type),
    FOREIGN KEY (match_id) REFERENCES public.match (match_id),
    FOREIGN KEY (player_id) REFERENCES public.player (player_id)	
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

CREATE TABLE public.batter_score (
    match_id VARCHAR(20) NOT NULL,
    over_num SMALLINT NOT NULL,
    innings_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    run_scored SMALLINT,
    type_run VARCHAR(20),
    PRIMARY KEY (match_id, innings_num, over_num, ball_num),
    FOREIGN KEY (match_id) REFERENCES public.match(match_id)
);

CREATE TABLE public.extras (
    match_id VARCHAR(20) NOT NULL,
    innings_num SMALLINT NOT NULL,
    over_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    extra_runs SMALLINT,
    extra_type VARCHAR(20),
    PRIMARY KEY (match_id, innings_num, over_num, ball_num),
    FOREIGN KEY (match_id) REFERENCES public.match(match_id)
);

CREATE TABLE public.match (
    match_id VARCHAR(20) NOT NULL,
    match_type VARCHAR(20),
    venue VARCHAR(20) NOT NULL,
    team_1_id VARCHAR(20) NOT NULL,
    team_2_id VARCHAR(20) NOT NULL,
    match_date DATE,
    season_id VARCHAR(20) NOT NULL,
    win_run_margin SMALLINT,
    win_by_wickets SMALLINT,
    win_type VARCHAR(20),
    toss_winner SMALLINT,
    toss_decide VARCHAR(20),
    winner_team_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (match_id),
    FOREIGN KEY (venue) REFERENCES public.team(region),
    FOREIGN KEY (team_1_id) REFERENCES public.team(team_id),
    FOREIGN KEY (team_2_id) REFERENCES public.team(team_id),
    FOREIGN KEY (season_id) REFERENCES public.season(season_id),
    FOREIGN KEY (winner_team_id) REFERENCES public.team(team_id)
);

CREATE TABLE public.player (
    player_id VARCHAR(20) NOT NULL,
    player_name VARCHAR(255),
    dob DATE,
    batting_hand VARCHAR(20),
    bowling_skill VARCHAR(20),
    country_name VARCHAR(20),
    PRIMARY KEY (player_id)
);

CREATE TABLE public.player_match (
    player_id VARCHAR(20) NOT NULL,
    match_id VARCHAR(20) NOT NULL,
    role VARCHAR(20),
    team_id VARCHAR(20) NOT NULL,
    is_extra BOOLEAN,
    PRIMARY KEY (player_id, match_id),
    FOREIGN KEY (player_id) REFERENCES public.player(player_id),
    FOREIGN KEY (match_id) REFERENCES public.match(match_id),
    FOREIGN KEY (team_id) REFERENCES public.team(team_id)
);

CREATE TABLE public.player_team (
    player_id VARCHAR(20) NOT NULL,
    team_id VARCHAR(20) NOT NULL,
    season_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (player_id, team_id, season_id)
);

CREATE TABLE public.season (
    season_id VARCHAR(20) NOT NULL,
    year SMALLINT,
    start_date DATE,
    end_date DATE,
    PRIMARY KEY (season_id)
);

CREATE TABLE public.team (
    team_id VARCHAR(20) NOT NULL,
    team_name VARCHAR(255),
    coach_name VARCHAR(255),
    region VARCHAR(20),
    PRIMARY KEY (team_id)
);

CREATE TABLE public.wickets (
    match_id VARCHAR(20) NOT NULL,
    innings_num SMALLINT NOT NULL,
    over_num SMALLINT NOT NULL,
    ball_num SMALLINT NOT NULL,
    player_out_id VARCHAR(20) NOT NULL,
    kind_out VARCHAR(20),
    fielder_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (match_id, innings_num, over_num, ball_num),
    FOREIGN KEY (match_id) REFERENCES public.match(match_id),
    FOREIGN KEY (player_out_id) REFERENCES public.player(player_id),
    FOREIGN KEY (fielder_id) REFERENCES public.player(player_id)
);
