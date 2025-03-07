package com.example;

import org.junit.jupiter.api.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.*;
import java.sql.*;
import java.util.stream.Collectors;
import java.util.HashMap;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

import com.example.DatabaseConnection;

public class PostgreSQLTest {
    private static Connection connection;
    // private static String DATABASE_NAME = "cricket_db";
    private static String DATABASE_NAME = System.getProperty("dataBase", "cricket_db").toLowerCase();
    private static String filePath = System.getProperty("schemaFile", "src/test/resources/schema.sql");

    static void executeSQLFile(Connection conn, String filePath) {
        try {
            String user_name = DatabaseConnection.get_user();
            String password = DatabaseConnection.get_password();
            String command = "sh src/test/resources/run.sh " + DATABASE_NAME + ' ' + user_name + ' ' + password + ' '
                    + filePath;
            Process process = Runtime.getRuntime().exec(command);

            // Read the output from the command
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            // System.out.println("Output:");
            while ((line = reader.readLine()) != null) {
                // System.out.println(line);
            }

            // Read any error messages from the command
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            // System.out.println("Errors:");
            while ((line = errorReader.readLine()) != null) {
                // System.err.println(line);
            }

            // Wait for the process to finish
            int exitCode = process.waitFor();
            // System.out.println("Exit Code: " + exitCode);

            System.out.println("Database created successfully!");

        } catch (Exception e) {
            System.out.println(e);
        }
    }

    @BeforeAll
    static void setup() throws SQLException {
        Connection main_conn = DatabaseConnection.getConnection("postgres");
        Statement main_stat = main_conn.createStatement();
        main_stat.execute("DROP DATABASE IF EXISTS " + DATABASE_NAME);
        main_stat.execute("CREATE DATABASE " + DATABASE_NAME);
        main_stat.close();

        connection = DatabaseConnection.getConnection(DATABASE_NAME);
        executeSQLFile(connection, filePath);
        connection = DatabaseConnection.getConnection(DATABASE_NAME);
    }

    @AfterAll
    static void tearDown() throws SQLException {
        if (connection != null) {
            connection.close();
        }

        Connection main_conn = DatabaseConnection.getConnection("postgres");
        Statement main_stat = main_conn.createStatement();

        main_stat.execute("SELECT pg_terminate_backend(pg_stat_activity.pid) " +
                "FROM pg_stat_activity " +
                "WHERE pg_stat_activity.datname = '" + DATABASE_NAME + "' " +
                "AND pid <> pg_backend_pid();");

        main_stat.execute("DROP DATABASE IF EXISTS " + DATABASE_NAME);
        main_conn.close();
    }

    @BeforeEach
    void resetDatabase() throws SQLException {
        try (Statement stmt = connection.createStatement()) {
        }
    }

    private void insertDependencies(String table, Statement stmt) throws SQLException {
        // Insert minimal required data for foreign keys
        if (table.equals("auction")) {
            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India')");
        }
        if (table.equals("awards")) {
            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Test Team 2', 'Test Coach 2', 'Test Region 2')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'playoff', 'Test Region', 1, 2, '2025-05-01', 'IPL2025')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India')");
        }
        if (table.equals("balls")) {
            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Test Team 2', 'Test Coach 2', 'Test Region 2')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (2, 'Test Player 2', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (3, 'Test Player 3', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'playoff', 'Test Region', 1, 2, '2025-05-01', 'IPL2025')");
        }
        if (table.equals("batter_score")) {
            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Test Team 2', 'Test Coach 2', 'Test Region 2')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (2, 'Test Player 2', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (3, 'Test Player 3', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'playoff', 'Test Region', 1, 2, '2025-05-01', 'IPL2025')");
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, 2, 3)");
        }
        if (table.equals("extras")) {
            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Test Team 2', 'Test Coach 2', 'Test Region 2')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (2, 'Test Player 2', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (3, 'Test Player 3', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'playoff', 'Test Region', 1, 2, '2025-05-01', 'IPL2025')");
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, 2, 3)");
        }
        if (table.equals("match")) {
            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Test Team 2', 'Test Coach 2', 'Test Region 2')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (2, 'Test Player 2', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (3, 'Test Player 3', '2000-01-01', 'right', 'India')");
        }
        if (table.equals("player")) {
        }
        if (table.equals("player_match")) {
            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Test Team 2', 'Test Coach 2', 'Test Region 2')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (2, 'Test Player 2', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (3, 'Test Player 3', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'playoff', 'Test Region', 1, 2, '2025-05-01', 'IPL2025')");
        }
        if (table.equals("player_team")) {
            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold) VALUES (1, 'IPL2025', 1, 1000000, 2000000, TRUE)");
        }
        if (table.equals("season")) {
        }
        if (table.equals("team")) {
        }
        if (table.equals("wickets")) {
            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Test Team 2', 'Test Coach 2', 'Test Region 2')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (2, 'Test Player 2', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (3, 'Test Player 3', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'playoff', 'Test Region', 1, 2, '2025-05-01', 'IPL2025')");
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, 2, 3)");
        }
    }

    // Generic test helper
    private boolean testNullConstraint(String table, String column, String insertQuery) {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            // Insert required dependencies
            insertDependencies(table, stmt);
            stmt.execute(insertQuery);

            return false;
        } catch (SQLException e) {
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            try {
                connection.rollback();
                connection.setAutoCommit(true);
            } catch (SQLException ignored) {
            }
        }
    }

    // Generic test helper with dependency handling
    private void testConstraint(String table, String column, String insertQuery, String expectedError) {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            // Insert required dependencies
            insertDependencies(table, stmt);
            stmt.execute(insertQuery);

            fail("Should throw exception for invalid " + column);
        } catch (SQLException e) {
            // assertTrue(e.getMessage().contains(expectedError),
            // "Failed constraint for " + column + ": " + e.getMessage());
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("check constraint"), "Failed check constraint for " + column + ": " + e.getMessage());
        } finally {
            try {
                connection.rollback();
                connection.setAutoCommit(true);
            } catch (SQLException ignored) {
            }
        }
    }

    // Generic test helper with dependency handling
    private boolean testPartialConstraint(String table, String column, String insertQuery) {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            // Insert required dependencies
            insertDependencies(table, stmt);
            stmt.execute(insertQuery);

            return false;
        } catch (SQLException e) {
            // "Failed constraint for " + column + ": " + e.getMessage());
            // System.out.println(e.getMessage());
            if(e.getMessage().toLowerCase().contains("check constraint")) return true;
            return false;
        } finally {
            try {
                connection.rollback();
                connection.setAutoCommit(true);
            } catch (SQLException ignored) {
            }
        }
    }

    private void testSanityCheck(String tableName) {
        try {
            DatabaseMetaData metaData = connection.getMetaData();
            ResultSet rs = metaData.getTables(null, null, tableName, new String[]{"TABLE"});
            
            boolean tableExists = rs.next(); // If there's at least one result, the table exists
            assertTrue(tableExists, "Table '" + tableName + "' should exist in the database!");
        } catch (AssertionError | SQLException e) {
            fail(e.getMessage());
        }
    }

    /************** Sanity Check **************/

    @Test
    @Tag("1_mark")
    void testSanity_Auction() throws Exception {
        testSanityCheck("auction");
    }

    @Test
    @Tag("1_mark")
    void testSanity_Awards() throws Exception {
        testSanityCheck("awards");
    }
    
    @Test
    @Tag("1_mark")
    void testSanity_Balls() throws Exception {
        testSanityCheck("balls");
    }

    @Test
    @Tag("1_mark")
    void testSanity_BatterScore() throws Exception {
        testSanityCheck("batter_score");
    }

    @Test
    @Tag("1_mark")
    void testSanity_Extras() throws Exception {
        testSanityCheck("extras");
    }

    @Test
    @Tag("1_mark")
    void testSanity_Match() throws Exception {
        testSanityCheck("match");
    }

    @Test
    @Tag("1_mark")
    void testSanity_Player() throws Exception {
        testSanityCheck("player");
    }

    @Test
    @Tag("1_mark")
    void testSanity_PlayerMatch() throws Exception {
        testSanityCheck("player_match");
    }

    @Test
    @Tag("1_mark")
    void testSanity_PlayerTeam() throws Exception {
        testSanityCheck("player_team");
    }

    @Test
    @Tag("1_mark")
    void testSanity_Season() throws Exception {
        testSanityCheck("season");
    }

    @Test
    @Tag("1_mark")
    void testSanity_Team() throws Exception {
        testSanityCheck("team");
    }

    @Test
    @Tag("1_mark")
    void testSanity_Wickets() throws Exception {
        testSanityCheck("wickets");
    }

    /************** Null Constraints **************/

    @Test
    @Tag("1_mark")
    void testNull_Auction() throws Exception {
        boolean testAuctionId = testNullConstraint("auction", "auction_id", "INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold) VALUES (null, 'IPL2025', 1, 10000000, false)");
        boolean testIsSold = testNullConstraint("auction", "is_sold", "INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold) VALUES (1, 'IPL2025', 1, 10000000, null)");
        boolean testSeasonId = testNullConstraint("auction", "season_id", "INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold) VALUES (1, null, 1, 10000000, false)");
        boolean testPlayerId = testNullConstraint("auction", "player_id", "INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold) VALUES (1, 'IPL2025', null, 10000000, false)");
        boolean testBasePrice = testNullConstraint("auction", "base_price", "INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold) VALUES (1, 'IPL2025', 1, null, false)");
        if (!testAuctionId || !testIsSold  || !testSeasonId || !testPlayerId || !testBasePrice) {
            fail("Auction Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_Awards() throws Exception {
        boolean testAwardType = testNullConstraint("awards", "award_type", "INSERT INTO awards (match_id, award_type, player_id) VALUES ('IPL2025001', null, 1)");
        boolean testMatchId = testNullConstraint("awards", "match_id", "INSERT INTO awards (match_id, award_type, player_id) VALUES (null, 'orange_cap', 1)");
        boolean testPlayerId = testNullConstraint("awards", "player_id", "INSERT INTO awards (match_id, award_type, player_id) VALUES ('IPL2025001', 'orange_cap', null)");
        if (!testAwardType || !testMatchId || !testPlayerId) {
            fail("Awards Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_Balls() throws Exception {
        boolean testMatchId = testNullConstraint("balls", "match_id", "INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES (null, 1, 1, 1, 1, 2, 3)");
        boolean testInningsNum = testNullConstraint("balls", "innings_num", "INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', null, 1, 1, 1, 2, 3)");
        boolean testOverNum = testNullConstraint("balls", "over_num", "INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, null, 1, 1, 2, 3)");
        boolean testBallNum = testNullConstraint("balls", "ball_num", "INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, null, 1, 2, 3)");
        boolean testStrikerId = testNullConstraint("balls", "striker_id", "INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, null, 2, 3)");
        boolean testNonStrikerId = testNullConstraint("balls", "non_striker_id", "INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, null, 3)");
        boolean testBowlerId = testNullConstraint("balls", "bowler_id", "INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, 2, null)");
        if (!testMatchId || !testInningsNum || !testOverNum || !testBallNum || !testStrikerId || !testNonStrikerId || !testBowlerId) {
            fail("Balls Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_BatterScore() throws Exception {
        boolean testMatchId = testNullConstraint("batter_score", "match_id", "INSERT INTO batter_score (match_id, over_num, innings_num, ball_num, run_scored, type_run) VALUES (null, 1, 1, 1, 1, 'running')");
        boolean testOverNum = testNullConstraint("batter_score", "over_num", "INSERT INTO batter_score (match_id, over_num, innings_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', null, 1, 1, 1, 'running')");
        boolean testInningsNum = testNullConstraint("batter_score", "innings_num", "INSERT INTO batter_score (match_id, over_num, innings_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, null, 1, 1, 'running')");
        boolean testBallNum = testNullConstraint("batter_score", "ball_num", "INSERT INTO batter_score (match_id, over_num, innings_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, null, 1, 'running')");
        boolean testRunScored = testNullConstraint("batter_score", "run_scored", "INSERT INTO batter_score (match_id, over_num, innings_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, 1, null, 'running')");
        if (!testMatchId || !testOverNum || !testInningsNum || !testBallNum || !testRunScored) {
            fail("Batter Score Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_Extras() throws Exception {
        boolean testMatchId = testNullConstraint("extras", "match_id", "INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES (null, 1, 1, 1, 'wide', 1)");
        boolean testInningsNum = testNullConstraint("extras", "innings_num", "INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', null, 1, 1, 'wide', 1)");
        boolean testOverNum = testNullConstraint("extras", "over_num", "INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', 1, null, 1, 'wide', 1)");
        boolean testBallNum = testNullConstraint("extras", "ball_num", "INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', 1, 1, null, 'wide', 1)");
        boolean testExtraType = testNullConstraint("extras", "extra_type", "INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', 1, 1, 1, null, 1)");
        boolean testExtraRuns = testNullConstraint("extras", "extra_runs", "INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', 1, 1, 1, 'wide', null)");
        if (!testMatchId || !testInningsNum || !testOverNum || !testBallNum || !testExtraType || !testExtraRuns) {
            fail("Extras Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_Match() throws Exception {
        boolean testMatchId = testNullConstraint("match", "match_id", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES (null, 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025')");
        boolean testMatchType = testNullConstraint("match", "match_type", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', null, 'Test Region', 1, 2, '2025-05-01', 'IPL2025')");
        boolean testVenue = testNullConstraint("match", "venue", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', null, 1, 2, '2025-05-01', 'IPL2025')");
        boolean testTeam1Id = testNullConstraint("match", "team_1_id", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', 'Test Region', null, 2, '2025-05-01', 'IPL2025')");
        boolean testTeam2Id = testNullConstraint("match", "team_2_id", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', 'Test Region', 1, null, '2025-05-01', 'IPL2025')");
        boolean testMatchDate = testNullConstraint("match", "match_date", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, null, 'IPL2025')");
        boolean testSeasonId = testNullConstraint("match", "season_id", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', null)");
        if (!testMatchId || !testMatchType || !testVenue || !testTeam1Id || !testTeam2Id || !testMatchDate || !testSeasonId) {
            fail("Match Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_Player() throws Exception {
        boolean testPlayerId = testNullConstraint("player", "player_id", "INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (null, 'Test Player', '2000-01-01', 'right', 'India')");
        boolean testPlayerName = testNullConstraint("player", "player_name", "INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, null, '2000-01-01', 'right', 'India')");
        boolean testDob = testNullConstraint("player", "dob", "INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', null, 'right', 'India')");
        boolean testBattingHand = testNullConstraint("player", "batting_hand", "INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', null, 'India')");
        boolean testCountryName = testNullConstraint("player", "country_name", "INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', null)");
        if (!testPlayerId || !testPlayerName || !testDob || !testBattingHand || !testCountryName) {
            fail("Player Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_PlayerMatch() throws Exception {
        boolean testMatchId = testNullConstraint("player_match", "match_id", "INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES (null, 1, 'batter', 1, false)");
        boolean testPlayerId = testNullConstraint("player_match", "player_id", "INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', null, 'batter', 1, false)");
        boolean testRole = testNullConstraint("player_match", "role", "INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 1, null, 1, false)");
        boolean testTeamId = testNullConstraint("player_match", "team_id", "INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 1, 'batter', null, false)");
        boolean testIsExtra = testNullConstraint("player_match", "is_extra", "INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 1, 'batter', 1, null)");
        if (!testMatchId || !testPlayerId || !testRole || !testTeamId || !testIsExtra) {
            fail("Player Match Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_PlayerTeam() throws Exception {
        boolean testPlayerId = testNullConstraint("player_team", "player_id", "INSERT INTO player_team (team_id, player_id, role) VALUES (1, null, 'batter')");
        boolean testTeamId = testNullConstraint("player_team", "team_id", "INSERT INTO player_team (team_id, player_id, role) VALUES (null, 1, 'batter')");
        boolean testRole = testNullConstraint("player_team", "role", "INSERT INTO player_team (team_id, player_id, role) VALUES (1, 1, null)");
        if (!testPlayerId || !testTeamId || !testRole) {
            fail("Player Team Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_Season() throws Exception {
        boolean testYear = testNullConstraint("season", "year", "INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, null, '2025-01-01', '2025-12-31')");
        boolean testStartDate = testNullConstraint("season", "start_date", "INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, null, '2025-12-31')");
        boolean testEndDate = testNullConstraint("season", "end_date", "INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', null)");
        if (!testYear || !testStartDate || !testEndDate) {
            fail("Season Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_Team() throws Exception {
        boolean testTeamId = testNullConstraint("team", "team_id", "INSERT INTO team (team_id, team_name, coach_name, region) VALUES (null, 'Test Team', 'Test Coach', 'Test Region')");
        boolean testTeamName = testNullConstraint("team", "team_name", "INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, null, 'Test Coach', 'Test Region')");
        boolean testCoachName = testNullConstraint("team", "coach_name", "INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', null, 'Test Region')");
        boolean testRegion = testNullConstraint("team", "region", "INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', null)");
        if (!testTeamId || !testTeamName || !testCoachName || !testRegion) {
            fail("Team Table null constraints not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testNull_Wickets() throws Exception {
        boolean testMatchId = testNullConstraint("wickets", "match_id", "INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES (null, 1, 1, 1, 1, 'bowled')");
        boolean testInningsNum = testNullConstraint("wickets", "innings_num", "INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', null, 1, 1, 1, 'bowled')");
        boolean testOverNum = testNullConstraint("wickets", "over_num", "INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', 1, null, 1, 1, 'bowled')");
        boolean testBallNum = testNullConstraint("wickets", "ball_num", "INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', 1, 1, null, 1, 'bowled')");
        boolean testPlayerOutId = testNullConstraint("wickets", "player_out_id", "INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', 1, 1, 1, null, 'bowled')");
        boolean testKindOut = testNullConstraint("wickets", "kind_out", "INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', 1, 1, 1, 1, null)");
        if (!testMatchId || !testInningsNum || !testOverNum || !testBallNum || !testPlayerOutId || !testKindOut) {
            fail("Wickets Table null constraints not satified!");
        }
    }

    /************** Additional Null Constraint Tests **************/

    boolean testAdditionalNull_Match_1() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'draw', 10, null)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testAdditionalNull_Match_2() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'draw', null, 4)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testAdditionalNull_Match_3() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'draw', null, null)");

            return true;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testAdditionalNull_Match_Draw() throws Exception {
        boolean result = testAdditionalNull_Match_1() && testAdditionalNull_Match_2() && testAdditionalNull_Match_3();
        if (!result) fail("Additional Null Constraints for Match Table when draw not maintained!");
    }

    boolean testAdditionalNull_Match_4() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'wickets', null, null)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testAdditionalNull_Match_5() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'wickets', 40, null)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testAdditionalNull_Match_6() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'wickets', null, 3)");

            return true;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testAdditionalNull_Match_7() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'wickets', 40, 4)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testAdditionalNull_Match_Wickets() throws Exception {
        boolean result = testAdditionalNull_Match_4() && testAdditionalNull_Match_5() && testAdditionalNull_Match_6() && testAdditionalNull_Match_7();
        if (!result) fail("Additional Null Constraints for Match Table when won by wickets not maintained!");
    }

    boolean testAdditionalNull_Match_8() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'runs', null, null)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testAdditionalNull_Match_9() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'runs', null, 3)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testAdditionalNull_Match_10() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'runs', 13, null)");

            return true;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testAdditionalNull_Match_11() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type, win_run_margin, win_by_wickets) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', 'runs', 40, 4)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testAdditionalNull_Match_Runs() throws Exception {
        boolean result = testAdditionalNull_Match_4() && testAdditionalNull_Match_5() && testAdditionalNull_Match_6() && testAdditionalNull_Match_7();
        if (!result) fail("Additional Null Constraints for Match Table when won by runs not maintained!");
    }

    boolean testSold_Auction_TeamId() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("auction", stmt);
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold, team_id, sold_price) VALUES (1, 'IPL2025', 1, 10000000, true, null, 20000000)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            // assertTrue(e.getMessage().toLowerCase().contains("for stumped dismissal, fielder must be a wicketkeeper"), "Failed: " + e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testSold_Auction_SoldPrice() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("auction", stmt);
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold, team_id, sold_price) VALUES (1, 'IPL2025', 1, 10000000, true, 1, null)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            // assertTrue(e.getMessage().toLowerCase().contains("for stumped dismissal, fielder must be a wicketkeeper"), "Failed: " + e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testSold_Aution() throws Exception {
        boolean result = testSold_Auction_SoldPrice() && testSold_Auction_TeamId();
        if (!result) fail("Additional Null Constraints when is_sold=true is not maintained!");
    }

    @Test
    @Tag("1_mark")
    void testSold_Auction_SoldPriceComp() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("auction", stmt);
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold, team_id, sold_price) VALUES (1, 'IPL2025', 1, 20000000, true, 1, 10000000)");

            fail("Sold Price is less than Base Price!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            // assertTrue(e.getMessage().toLowerCase().contains("null"), "Failed: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testCaught_FielderRole() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("wickets", stmt);
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 3, null, 1, FALSE)");
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out, fielder_id) VALUES ('IPL2025001', 1, 1, 1, 1, 'caught', 3)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            // assertTrue(e.getMessage().toLowerCase().contains("for stumped dismissal, fielder must be a wicketkeeper"), "Failed: " + e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testRunOut_FielderRole() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("wickets", stmt);
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 3, null, 1, FALSE)");
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out, fielder_id) VALUES ('IPL2025001', 1, 1, 1, 1, 'runout', 3)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            // assertTrue(e.getMessage().toLowerCase().contains("for stumped dismissal, fielder must be a wicketkeeper"), "Failed: " + e.getMessage());
            if (e.getMessage().toLowerCase().contains("null")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testCaughtRunOut_FielderRole() throws Exception {
        boolean result = testCaught_FielderRole() && testRunOut_FielderRole();
        if (!result) fail("Caught/RunOut Fielder Role not checked!");
    }

    @Test
    @Tag("1_mark")
    void testStumped_FielderRole() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("wickets", stmt);
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 3, 'bowler', 1, FALSE)");
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out, fielder_id) VALUES ('IPL2025001', 1, 1, 1, 1, 'stumped', 3)");

            fail("Stumped Fielder Role not checked!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("for stumped dismissal, fielder must be a wicketkeeper"), "Failed: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    /************** Allowed Value Constraint Tests **************/

    @Test
    @Tag("1_mark")
    void testAllowed_Extras_ExtraType() throws Exception {
        testConstraint("extras", "extra_type",
                "INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_runs, extra_type) VALUES ('IPL2025001', 1, 1, 1, 1, 'invalid_extra')",
                "Allowed values: no_ball, wide, byes, legbyes");
    }

    @Test
    @Tag("1_mark")
    void testAllowed_Awards_AwardType() throws Exception {
        testConstraint("awards", "award_type",
                "INSERT INTO awards (match_id, award_type, player_id) " +
                        "VALUES ('IPL2025001', 'invalid_award', 1)",
                "Allowed values: orange_cap, purple_cap");
    }

    @Test
    @Tag("1_mark")
    void testAllowed_BatterScore_TypeRun() throws Exception {
        testConstraint("batter_score", "type_run",
                "INSERT INTO batter_score (match_id, over_num, innings_num, ball_num, run_scored, type_run) " +
                        "VALUES ('IPL2025001', 1, 1, 1, 1, 'invalid_run')",
                "Allowed values: running, boundary");
    }

    @Test
    @Tag("1_mark")
    void testAllowed_Match() throws Exception {
        boolean match_type = testPartialConstraint("match", "match_type", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'invalid_match', 'Test Region', 1, 2, '2025-05-01', 'IPL2025')");
        boolean win_type = testPartialConstraint("match", "win_type", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_type) VALUES ('IPL2025001', 'playoff', 'Test Region', 1, 2, '2025-01-01', 'IPL2025', 'invalid_win')");
        boolean toss_winner = testPartialConstraint("match", "toss_winner", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, toss_winner, toss_decide) VALUES ('IPL2025001', 'playoff', 'Test Region', 1, 2, '2025-01-01', 'IPL2025', 0, 'bat')");
        boolean toss_decision = testPartialConstraint("match", "toss_decide", "INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, toss_winner, toss_decide) VALUES ('IPL2025001', 'playoff', 'Test Region', 1, 2, '2025-01-01', 'IPL2025', 1, 'invalid_decision')");
        if (!match_type || !win_type || !toss_winner || !toss_decision) {
            fail("Match Table allowed values not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testAllowed_Player() throws Exception {
        boolean batting_hand = testPartialConstraint("player", "batting_hand", "INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'invalid_hand', 'India')");
        boolean bowling_skill = testPartialConstraint("player", "bowling_skill", "INSERT INTO player (player_id, player_name, dob, batting_hand, country_name, bowling_skill) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India', 'invalid_skill')");
        if (!batting_hand || !bowling_skill) {
            fail("Player Table allowed values not satified!");
        }
    }

    @Test
    @Tag("1_mark")
    void testAllowed_PlayerMatch_Role() throws Exception {
        testConstraint("player_match", "role",
                "INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) " +
                        "VALUES ('IPL2025001', 1, 'invalid_role', 1, false)",
                "Allowed values: batter, bowler, allrounder, wicketkeeper");
    }

    @Test
    @Tag("1_mark")
    void testAllowed_Wickets_KindOut() throws Exception {
        testConstraint("wickets", "kind_out",
                "INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) " +
                        "VALUES ('IPL2025001', 1, 1, 1, 1, 'invalid_kind')",
                "Allowed values: bowled, caught, lbw, runout, stumped, hitwicket");
    }

    @Test
    @Tag("1_mark")
    void testBatterScore_RunScored() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO batter_score (match_id, over_num, innings_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, 1, -2, 'running')");

            fail("Run Scored constraint not satified!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("check constraint"), "Failed check constraint : " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testExtras_ExtraRuns() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("extras", stmt);
            stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', 1, 1, 1, 'wide', -1)");

            fail("Extra Runs constraint not satified!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("check constraint"), "Failed check constraint : " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testPlayer_DOB() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Rashmi', '2016-02-03', 'right', 'India')");

            fail("Player DOB constraint not satified!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("check constraint"), "Failed check constraint : " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testSeason_Year() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2026, '2026-01-01', '2026-12-31')");

            fail("Run Scored constraint not satified!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("check constraint"), "Failed check constraint : " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testAuction_BasePrice() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("auction", stmt);
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold) VALUES (1, 'IPL2025', 1, 999999, false)");

            fail("Base Price Constraint not satisfied!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("check constraint"), "Failed check constraint : " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    /************** Foreign Key Constraint Tests **************/

    boolean testForeign_Auction_SeasonId() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("auction", stmt);
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold, sold_price, team_id) VALUES (1, 'IPL2020', 1, 1000000, TRUE, 2000000, 1)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testForeign_Auction_PlayerId() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("auction", stmt);
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold, sold_price, team_id) VALUES (1, 'IPL2025', 2, 1000000, TRUE, 2000000, 1)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testForeign_Auction_TeamId() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("auction", stmt);
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold, sold_price, team_id) VALUES (1, 'IPL2025', 1, 1000000, TRUE, 2000000, 2)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testForeign_Auction() throws Exception {
        boolean result = testForeign_Auction_SeasonId() && testForeign_Auction_PlayerId() && testForeign_Auction_TeamId();
        if (!result) fail("Foreign Key Violation in auction not flagged!");
    }

    boolean testForeign_Awards_PlayerID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("awards", stmt);
            stmt.execute("INSERT INTO awards (match_id, award_type, player_id) VALUES ('IPL2025001', 'orange_cap', 'INVALID_PLAYER')");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    };

    boolean testForeign_Awards_MatchID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("awards", stmt);
            stmt.execute("INSERT INTO awards (match_id, award_type, player_id) VALUES ('INVALID_MATCH', 'orange_cap', '1')");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    @Test
    @Tag("1_mark")
    void testForeign_Awards() throws Exception {
        boolean result = testForeign_Awards_PlayerID() && testForeign_Awards_MatchID();
        if (!result) fail("Foreign Key Violation in awards not flagged!");
    }

    boolean testForeign_Balls_MatchID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("balls", stmt);
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('INVALID_MATCH', 1, 1, 1, 1, 2, 3)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    boolean testForeign_Balls_StrikerID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("balls", stmt);
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 'INVALID_PLAYER', 2, 3)");
            
            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    boolean testForeign_Balls_NonStrikerID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("balls", stmt);
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, 'INVALID_PLAYER', 3)");
            
            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    boolean testForeign_Balls_BowlerID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("balls", stmt);
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, 2, 'INVALID_PLAYER')");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    @Test
    @Tag("1_mark")
    void testForeign_Balls() throws Exception {
        boolean result = testForeign_Balls_MatchID() && testForeign_Balls_StrikerID() && testForeign_Balls_NonStrikerID() && testForeign_Balls_BowlerID();
        if (!result) fail("Foreign Key Violation in balls not flagged!");
    }

    boolean testForeign_BatterScore_MatchID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored) VALUES ('IPL2025001', 1, 1, 1, 0)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    boolean testForeign_BatterScore_All() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("balls", stmt);
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored) VALUES ('IPL2025001', 1, 1, 2, 0)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    @Test
    @Tag("1_mark")
    void testForeign_BatterScore() throws Exception {
        boolean result = testForeign_BatterScore_MatchID() && testForeign_BatterScore_All();
        if (!result) fail("Foreign Key Violation in batter_score not flagged!");
    }

    boolean testForeign_Extras_MatchID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_runs, extra_type) VALUES ('INVALID_MATCH', 1, 1, 1, 5, 'wide')");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    boolean testForeign_Extras_All() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("balls", stmt);
            stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_runs, extra_type) VALUES ('IPL2025001', 1, 2, 1, 5, 'wide')");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    @Test
    @Tag("1_mark")
    void testForeign_Extras() throws Exception {
        boolean result = testForeign_Extras_MatchID() && testForeign_Extras_All();
        if (!result) fail("Foreign Key Violation in extras not flagged!");
    }

    boolean testForeign_Match_Venue() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'playoff', 'Test Venue', 1, 2, '2025-05-15', 'IPL2025')");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    boolean testForeign_Match_Team1() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', 'Test Region', 3, 2, '2025-05-15', 'IPL2025')");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    boolean testForeign_Match_Team2() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', 'Test Region', 1, 3, '2025-05-15', 'IPL2025')");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    boolean testForeign_Match_SeasonID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('1001', 'league', 'Test Region', 1, 2, '2025-05-15', 1)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    @Test
    @Tag("1_mark")
    void testForeign_Match() throws Exception {
        boolean result = testForeign_Match_Venue() && testForeign_Match_Team1() && testForeign_Match_Team2() && testForeign_Match_SeasonID();
        if (!result) fail("Foreign Key Violation in match not flagged!");
    }

    boolean testForeign_PlayerMatch_PlayerID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("player_match", stmt);
            stmt.execute("INSERT INTO player_match (player_id, match_id, role, team_id, is_extra) VALUES (4, 'IPL2025001', 'batter', 1, FALSE)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testForeign_PlayerMatch_MatchID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("player_match", stmt);
            stmt.execute("INSERT INTO player_match (player_id, match_id, role, team_id, is_extra) VALUES (1, 8, 'batter', 1, FALSE)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testForeign_PlayerMatch_TeamID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("player_match", stmt);
            stmt.execute("INSERT INTO player_match (player_id, match_id, role, team_id, is_extra) VALUES (2, 'IPL2025001', 'batter', 10, FALSE)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("1_mark")
    void testForeign_PlayerMatch() throws Exception {
        boolean result = testForeign_PlayerMatch_PlayerID() && testForeign_PlayerMatch_MatchID() && testForeign_PlayerMatch_TeamID();
        if (!result) fail("Foreign Key Violation in player_match not flagged!");
    }

    @Test
    @Tag("1_mark")
    void testForeign_PlayerTeam() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (999, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (999, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (999, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player_team (player_id, team_id, season_id) VALUES (999, 999, 'IPL2025')");

            fail("Foreign Key Violation in player_match not flagged!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("foreign key"), "Failed foreign key constraint : " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    boolean testForeign_Wickets_MatchID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("wickets", stmt);
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('INVALID_MATCH', 1, 1, 1, 1, 'bowled')");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    boolean testForeign_Wickets_PlayerOutID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("wickets", stmt);
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', 1, 1, 1, 'INVALID_PLAYER1', 'bowled')");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    boolean testForeign_Wickets_FielderID() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("wickets", stmt);
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out, fielder_id) VALUES ('IPL2025001', 1, 1, 1, 1, 'bowled', 4)");

            return false;
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            if (e.getMessage().toLowerCase().contains("foreign key")) return true;
            return false;
        } finally {
            connection.rollback();
            connection.setAutoCommit(true);
        }
    }

    @Test
    @Tag("1_mark")
    void testForeign_Wickets() throws Exception {
        boolean result = testForeign_Wickets_MatchID() && testForeign_Wickets_PlayerOutID() && testForeign_Wickets_FielderID();
        if (!result) fail("Foreign Key Violation in wickets not flagged!");
    }

    /************** Unique Constraint Tests **************/

    @Test
    @Tag("1_mark")
    void testTeam_Unique_TeamName() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Test Team', 'Test Coach 2', 'Test Region 2')");

            fail("Should throw exception for duplicate team name");
        } catch (SQLException e) {
            assertTrue(e.getMessage().toLowerCase().contains("unique"), "Failed unique constraint for team_name: " + e.getMessage());
        } finally {
            try {
                connection.rollback();
                connection.setAutoCommit(true);
            } catch (SQLException ignored) {
            }
        }
    }

    @Test
    @Tag("1_mark")
    void testTeam_Unique_Region() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Test Team 2', 'Test Coach 2', 'Test Region')");

            fail("Should throw exception for duplicate region");
        } catch (SQLException e) {
            assertTrue(e.getMessage().toLowerCase().contains("unique"), "Failed unique constraint for region: " + e.getMessage());
        } finally {
            try {
                connection.rollback();
                connection.setAutoCommit(true);
            } catch (SQLException ignored) {
            }
        }
    }

    @Test
    @Tag("1_mark")
    void testAuction_Unique() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Test Team', 'Test Coach', 'Test Region')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (1, 'IPL2025', 1, 10000000, 20000000, true, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (1, 'IPL2025', 1, 10000000, 14000000, true, 1)");

            fail("Should throw exception for duplicate auction_id");
        } catch (SQLException e) {
            assertTrue(e.getMessage().toLowerCase().contains("unique"), "Failed unique constraint in auction table: " + e.getMessage());
        } finally {
            try {
                connection.rollback();
                connection.setAutoCommit(true);
            } catch (SQLException ignored) {
            }
        }
    }

    /************** Miscellaneous Constraint Tests **************/

    @Test
    @Tag("3_mark")
    void testSeasonIDCalculator() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES (1, 2025, '2025-01-01', '2025-12-31')");

            ResultSet resultSet = stmt.executeQuery("SELECT season_id FROM season");

            if (resultSet.next()) {
                assertEquals(resultSet.getString(1), "IPL2025");
            } else {
                fail("No data found");
            }

        } catch (AssertionError | SQLException e) {
            fail(e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testMatchIDValidator() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025002', 'league', 'Test Region', 1, 2, '2025-05-15', 'IPL2025')");

            fail("Stumped Fielder Role not checked!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("sequence of match id violated"), "Failed: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testMatchIDFormat() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES (1, 'league', 'Test Region', 1, 2, '2025-05-15', 'IPL2025')");

            fail("Stumped Fielder Role not checked!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("sequence of match id violated"), "Failed: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testHome_Away_for_League() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Rashmi', '2000-02-03', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name, bowling_skill) VALUES (2, 'Maria', '1997-07-21', 'left', 'England', 'medium')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name, bowling_skill) VALUES (3, 'Govind', '2005-11-25', 'right', 'India', 'medium')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (4, 'Mark', '2002-09-07', 'left', 'Austalia')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Kerala Warriors', 'Subramanian', 'Thiruvananthapuram')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Kashmir Leopards', 'Lahar', 'Poonch')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (3, 'Arunachal Stormbreakers', 'Tsering', 'Tawang')");
            stmt.execute("INSERT INTO season (year, start_date, end_date) VALUES (2025, '2020-05-13', '2020-07-31')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', 'Tawang', 2, 1, '2020-06-12', 'IPL2025')");

            fail("Home/Away constraint for league mathces not checked!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("league match must be played at home ground of one of the teams"), "Failed: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testWinnerIDCalculator() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Rashmi', '2000-02-03', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name, bowling_skill) VALUES (2, 'Maria', '1997-07-21', 'left', 'England', 'medium')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name, bowling_skill) VALUES (3, 'Govind', '2005-11-25', 'right', 'India', 'medium')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (4, 'Mark', '2002-09-07', 'left', 'Austalia')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Kerala Warriors', 'Subramanian', 'Thiruvananthapuram')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Kashmir Leopards', 'Lahar', 'Poonch')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (3, 'Arunachal Stormbreakers', 'Tsering', 'Tawang')");
            stmt.execute("INSERT INTO season (year, start_date, end_date) VALUES (2025, '2020-05-13', '2020-07-31')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', 'Poonch', 2, 3, '2020-06-12', 'IPL2025')");
            stmt.execute("UPDATE match SET win_run_margin = 29, win_type = 'runs', toss_winner = 1, toss_decide = 'bowl' WHERE match_id = 'IPL2025001'");

            ResultSet resultSet = stmt.executeQuery("SELECT winner_team_id FROM match WHERE match_id = 'IPL2025001'");

            if (resultSet.next()) {
                assertEquals(resultSet.getString(1), "3");
            } else {
                fail("No data found");
            }

        } catch (AssertionError | SQLException e) {
            fail(e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testLeagueMatchCount() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Rashmi', '2000-02-03', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name, bowling_skill) VALUES (2, 'Maria', '1997-07-21', 'left', 'England', 'medium')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name, bowling_skill) VALUES (3, 'Govind', '2005-11-25', 'right', 'India', 'medium')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (4, 'Mark', '2002-09-07', 'left', 'Austalia')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Kerala Warriors', 'Subramanian', 'Thiruvananthapuram')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Kashmir Leopards', 'Lahar', 'Poonch')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (3, 'Arunachal Stormbreakers', 'Tsering', 'Tawang')");
            stmt.execute("INSERT INTO season (year, start_date, end_date) VALUES (2025, '2020-05-13', '2020-07-31')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', 'Poonch', 2, 3, '2020-06-12', 'IPL2025')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025002', 'league', 'Tawang', 2, 3, '2020-07-12', 'IPL2025')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025003', 'league', 'Poonch', 3, 2, '2020-06-12', 'IPL2025')");

            fail("Number of league matches not contrained!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("each team can play only one home match in a league against another team"), "Failed: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testInternationalPlayerCount() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (1, 'Rashmi', '2000-02-03', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name, bowling_skill) VALUES (2, 'Maria', '1997-07-21', 'left', 'England', 'legspin')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name, bowling_skill) VALUES (3, 'Gurpreet', '2005-11-25', 'right', 'India', 'medium')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (4, 'Mark', '2002-09-07', 'left', 'Austalia')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name, bowling_skill) VALUES (5, 'James', '1997-07-21', 'left', 'West Indies', 'offspin')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (6, 'Ann', '2005-11-25', 'right', 'Germany')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (1, 'Kerala Warriors', 'Subramanian', 'Thiruvananthapuram')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (2, 'Kashmir Leopards', 'Lahar', 'Poonch')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES (3, 'Arunachal Stormbreakers', 'Tsering', 'Tawang')");
            stmt.execute("INSERT INTO season (year, start_date, end_date) VALUES (2025, '2020-05-13', '2020-07-31')");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (1, 'IPL2025', 1, 2000000, 3000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (2, 'IPL2025', 2, 2000000, 3000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (3, 'IPL2025', 4, 2000000, 3000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (4, 'IPL2025', 5, 2000000, 3000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (5, 'IPL2025', 6, 2000000, 3000000, TRUE, 1)");

            fail("Number of international players not checked!");
        } catch (SQLException e) {
            // Expected exception, test should pass
            // System.out.println(e.getMessage());
            assertTrue(e.getMessage().toLowerCase().contains("there could be atmost 3 international players per team per season"), "Failed: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testAutomaticPlayerTeamInsertion() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            insertDependencies("auction", stmt);
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, is_sold, sold_price, team_id) VALUES (1, 'IPL2025', 1, 10000000, true, 20000000, 1)");

            ResultSet resultSet = stmt.executeQuery("SELECT COUNT(*) FROM player_team WHERE player_id = '1' AND team_id = '1' AND season_id = 'IPL2025'");
            if (resultSet.next()) {
                assertTrue(resultSet.getInt(1) > 0, "Entry (1, 1, 'IPL2025') is not present in player_team table.");
            } else {
                fail("Query execution failed.");
            }
        } catch (AssertionError | SQLException e) {
            fail(e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    // Testing Views
    // Batter stats
    @Test
    @Tag("3_mark")
    void testBatterView() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            // Insert dependencies
            insertDependencies("balls", stmt);
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (1, 'IPL2025', 1, 1000000, 2000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (2, 'IPL2025', 2, 1000000, 2000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (3, 'IPL2025', 3, 1000000, 2000000, TRUE, 2)");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025002', 'playoff', 'Test Region 2', 1 , 2 , '2025-05-01', 'IPL2025')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025003', 'playoff', 'Test Region', 1 , 2 , '2025-05-01', 'IPL2025')");
            // stmt.execute("INSERT INTO player_team (player_id, team_id, season_id) VALUES (1, 1, 'IPL2025')");
            // stmt.execute("INSERT INTO player_team (player_id, team_id, season_id) VALUES (2, 1, 'IPL2025')");
            // stmt.execute("INSERT INTO player_team (player_id, team_id, season_id) VALUES (3, 2, 'IPL2025')");

            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 1, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025002', 1, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025003', 1, 'batter', 1, FALSE)");

            
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 2, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025002', 2, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025003', 2, 'batter', 1, FALSE)");

            
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 3, 'batter', 2, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025002', 3, 'batter', 2, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025003', 3, 'batter', 2, FALSE)");

            // Insert balls and batter_score for match 1
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, 1, 4, 'boundary')");

            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 2, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, 2, 1, 'running')");

            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 3, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, 3, 2, 'running')");

            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 2, 1, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 2, 1, 6, 'boundary')");

            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 2, 2, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 2, 2, 2, 'running')");

            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 2, 3, 1, 2, 3)");
            // 15 runs in above match , 6 balls, 2 boundaries, 0 wicket

            // Insert balls and batter_score for match 2
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025002', 1, 1, 1, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025002', 1, 1, 1, 1, 'running')");

            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025002', 1, 1, 2, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025002', 1, 1, 2, 1, 'running')");

            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025002', 1, 1, 3, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025002', 1, 1, 3, 4, 'boundary')");

            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025002', 1, 1, 4, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025002', 1, 1, 4, 6, 'boundary')");

            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025002', 1, 1, 5, 1, 2, 3)");
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025002', 1, 1, 5, 1, 'bowled')");
            // 12 runs in above match , 5 balls, 2 boundaries, 1 wicket

            // Insert balls and batter_score for match 3
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025003', 1, 1, 1, 1, 2, 3)");
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025003', 1, 1, 1, 1, 'bowled')");
            // 1 wicket , 1 ball , 0 runs

            // Check if views are created
            ResultSet rs = stmt 
                    .executeQuery("SELECT table_name FROM information_schema.views WHERE table_name = 'batter_stats'");
            assertTrue(rs.next(), "batter_stats view is not created");

            // Check if the columns exist with correct values
            rs = stmt.executeQuery("SELECT * FROM batter_stats WHERE player_id = '1'");
            assertTrue(rs.next(), "No data found for player_id 1");

            assertEquals(rs.getString("player_id"), "1", "Incorrect player_id");
            assertEquals(rs.getInt("Mat"), 3, "Incorrect Mat");
            assertEquals(rs.getInt("Inns"), 3, "Incorrect Inns");
            assertEquals(rs.getInt("R"), 27, "Incorrect R"); 
            assertEquals(rs.getInt("HS"), 15, "Incorrect HS");
            assertEquals(rs.getDouble("Avg"), 13.5, "Incorrect Avg");
            assertEquals(rs.getDouble("SR"), 225, "Incorrect SR");
            assertEquals(rs.getInt("100s"), 0, "Incorrect 100s");
            assertEquals(rs.getInt("50s"), 0, "Incorrect 50s");
            assertEquals(rs.getInt("Ducks"), 1, "Incorrect Ducks");
            assertEquals(rs.getInt("BF"), 12, "Incorrect BF");
            assertEquals(rs.getInt("Boundaries"), 4, "Incorrect Boundaries");
            assertEquals(rs.getInt("NO"), 1, "Incorrect NO");

        } catch (AssertionError | SQLException e) {
            fail("Exception occurred: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testBowlerStatsView() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            // Insert dependencies
            insertDependencies("balls", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025002', 'playoff', 'Test Region', 1 , 2 , '2025-05-01', 'IPL2025')");
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025003', 'playoff', 'Test Region 2', 1 , 2 , '2025-05-01', 'IPL2025')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (4, 'Test Player 4', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (1, 'IPL2025', 1, 1000000, 2000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (2, 'IPL2025', 2, 1000000, 2000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (3, 'IPL2025', 3, 1000000, 2000000, TRUE, 2)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (4, 'IPL2025', 4, 1000000, 2000000, TRUE, 1)");

            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 1, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025002', 1, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025003', 1, 'batter', 1, FALSE)");

            
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 2, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025002', 2, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025003', 2, 'batter', 1, FALSE)");

            
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 3, 'batter', 2, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025002', 3, 'batter', 2, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025003', 3, 'batter', 2, FALSE)");

            
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 4, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025002', 4, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025003', 4, 'batter', 1, FALSE)");

            // Insert balls, batter_score, wickets, and extras for match 1
            for (int i = 1; i <= 8; i++) {
                if( i <= 4 )
                {
                    stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, " + i + ", 1, 2, 3)");
                }
                if( i > 4 )
                {
                    stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, " + i + ", 4, 2, 3)");
                }
                if (i %2 ==  0 && i <= 3) {
                    stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, " + i + ", " + (i % 4) + ", 'running')");
                }
                if (i %2 ==  1 && i > 4 && i <= 7) {
                    stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, " + i + ", " + (i % 4) + ", 'running')");
                }
                if (i == 4 ) {
                    stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', 1, 1, " + i + ", 1, 'bowled')");
                }
                if(i == 8) {
                    stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', 1, 1, " + i + ", 4, 'bowled')");
                }
                if (i == 6) {
                    stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', 1, 1, " + i + ", 'no_ball', 1)");
                }
                if (i == 7) {
                    stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', 1, 1, " + i + ", 'no_ball', 1)");
                }
            }
            // runs = 0 , 2  ,1 ,3   = 6 , extras = 2 


            // Insert balls, batter_score, wickets, and extras for match 2
            for (int i = 1; i <= 6; i++) {
                stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025002', 1, 1, " + i + ", 1, 2, 3)");
                if (i % 2 == 1) {
                    stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025002', 1, 1, " + i + ", " + (i % 4) + ", 'running')");
                }
                if (i == 6) {
                    stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025002', 1, 1, " + i + ", 1, 'bowled')");
                }
                if (i == 5) {
                    stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025002', 1, 1, " + i + ", 'no_ball', 1)");
                }
            }
            // runs = 1 , 3 , 1 = 5 , extras = 1 

            // Insert balls, batter_score, wickets, and extras for match 3
            for (int i = 1; i <= 7; i++) {
                stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025003', 1, 1, " + i + ", 1, 2, 3)");
                if (i % 2 == 0) {
                    stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025003', 1, 1, " + i + ", " + (i % 4) + ", 'running')");
                }
                if (i == 7) {
                    stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025003', 1, 1, " + i + ", 1, 'bowled')");
                }
                if (i == 1) {
                    stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025003', 1, 1, " + i + ", 'no_ball', 1)");
                }
                if (i == 2) {
                    stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025003', 1, 1, " + i + ", 'no_ball', 1)");
                }
            }
            // runs = 2, 2 = 4  , extras = 2

            // Check if views are created
            ResultSet rs = stmt
                    .executeQuery("SELECT table_name FROM information_schema.views WHERE table_name = 'bowler_stats'");
            assertTrue(rs.next(), "bowler_stats view is not created");

            // Check if the columns exist with correct values
            rs = stmt.executeQuery("SELECT * FROM bowler_stats WHERE player_id = '3'");
            assertTrue(rs.next(), "No data found for player_id 3");
            assertEquals(rs.getString("player_id"), "3", "Incorrect player_id");
            assertEquals(rs.getInt("B"), 21, "Incorrect Balls");
            assertEquals(rs.getInt("W"), 4, "Incorrect Wkts");
            assertEquals(rs.getInt("Runs"), 20, "Incorrect Runs");
            assertEquals(rs.getDouble("Avg"), 5, "Incorrect Avg");
            assertEquals(rs.getInt("Econ"), 6, "Incorrect Econ");
            assertEquals(rs.getDouble("SR"), 5.25, "Incorrect SR");
            assertEquals(rs.getInt("Extras"), 5, "Incorrect extras");

        } catch (AssertionError | SQLException e) {
            fail("Exception occurred: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }
    
    @Test
    @Tag("3_mark")
    void testFielderStatsView() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();
            insertDependencies("balls", stmt);
            
            // Insert players
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES "
                    + "(4, 'Test Player 4', '2000-01-01', 'right', 'India'),"
                    + "(5, 'Test Player 5', '2000-01-01', 'right', 'India'),"
                    + "(6, 'Test Player 6', '2000-01-01', 'right', 'India'),"
                    + "(7, 'Test Player 7', '2000-01-01', 'right', 'India'),"
                    + "(8, 'Test Player 8', '2000-01-01', 'right', 'India')");
            
            for (int i = 1; i <= 8; i++) {
                stmt.execute(String.format(
                    "INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) " +
                    "VALUES (%d, 'IPL2025', %d, 1000000, 2000000, TRUE, %d)", i, i, (i % 2) + 1));
            }
            
            // Insert match players
            for (int i = 1; i <= 8; i++) {
                if(i!=8)
                stmt.execute(String.format(
                    "INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) " +
                    "VALUES ('IPL2025001', %d, 'batter', %d, FALSE)", i, (i % 2) + 1));
                else 
                {
                    stmt.execute(String.format(
                    "INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) " +
                    "VALUES ('IPL2025001', %d, 'wicketkeeper', %d, FALSE)", i, (i % 2) + 1));
                }
            }
            
            // Insert balls data
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES "
                    + "('IPL2025001', 1, 1, 1, 1, 2, 3),"
                    + "('IPL2025001', 1, 1, 2, 5, 2, 3),"
                    + "('IPL2025001', 1, 1, 3, 6, 2, 3),"
                    + "('IPL2025001', 1, 1, 5, 7, 2, 3)");
            
            // Insert wickets data
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out, fielder_id) VALUES "
                    + "('IPL2025001', 1, 1, 1, 1, 'caught', 4),"
                    + "('IPL2025001', 1, 1, 2, 5, 'runout', 4),"
                    + "('IPL2025001', 1, 1, 3, 6, 'caught', 4),"
                    + "('IPL2025001', 1, 1, 5, 7, 'stumped', 8)");
            
            // Verify fielder_stats view exists
            ResultSet rs = stmt.executeQuery("SELECT table_name FROM information_schema.views WHERE table_name = 'fielder_stats'");
            assertTrue(rs.next(), "fielder_stats view is not created");
            
            // Verify data in fielder_stats
            rs = stmt.executeQuery("SELECT * FROM fielder_stats WHERE player_id = '4'");
            assertTrue(rs.next(), "No data found for player_id 4");
            assertEquals("4", rs.getString("player_id"), "Incorrect player_id");
            assertEquals(2, rs.getInt("C"), "Incorrect C");
            assertEquals(0, rs.getInt("St"), "Incorrect St");
            assertEquals(1, rs.getInt("RO"), "Incorrect RO");
            
            rs = stmt.executeQuery("SELECT * FROM fielder_stats WHERE player_id = '8'");
            assertTrue(rs.next(), "No data found for player_id 8");
            assertEquals("8", rs.getString("player_id"), "Incorrect player_id");
            assertEquals(0, rs.getInt("C"), "Incorrect C");
            assertEquals(1, rs.getInt("St"), "Incorrect St");
            assertEquals(0, rs.getInt("RO"), "Incorrect RO");
            
        } catch (AssertionError | SQLException e) {
            fail("Exception occurred: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testAuctionDeletion() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            // Insert dependencies
            insertDependencies("awards", stmt);
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (2, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (3, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (1, 'IPL2025', 1, 1000000, 2000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (2, 'IPL2025', 2, 1000000, 2000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (3, 'IPL2025', 3, 1000000, 2000000, TRUE, 2)");

            // Insert related records
            // stmt.execute("INSERT INTO player_team (player_id, team_id, season_id) VALUES (1, 1, 'IPL2025')");
            stmt.execute("INSERT INTO awards (match_id, award_type, player_id) VALUES ('IPL2025001', 'orange_cap', 1)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 1, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, 1, 4, 'boundary')");
            stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', 1, 1, 1, 'wide', 1)");
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', 1, 1, 1, 1, 'bowled')");

            // Delete auction record
            stmt.execute("DELETE FROM auction WHERE auction_id = '1'");

            // Verify related records are deleted
            ResultSet rs = stmt.executeQuery("SELECT * FROM player_team WHERE player_id = '1' AND season_id = 'IPL2025'");
            assertFalse(rs.next(), "player_team record not deleted");

            rs = stmt.executeQuery("SELECT * FROM awards WHERE player_id = '1'");
            assertFalse(rs.next(), "awards record not deleted");

            rs = stmt.executeQuery("SELECT * FROM player_match WHERE player_id = '1'");
            assertFalse(rs.next(), "player_match record not deleted");

            rs = stmt.executeQuery("SELECT * FROM balls WHERE striker_id = '1' OR non_striker_id = '1' OR bowler_id = '1'");
            assertFalse(rs.next(), "balls record not deleted");

            rs = stmt.executeQuery("SELECT * FROM batter_score WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "batter_score record not deleted");

            rs = stmt.executeQuery("SELECT * FROM extras WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "extras record not deleted");

            rs = stmt.executeQuery("SELECT * FROM wickets WHERE player_out_id = '1'");
            assertFalse(rs.next(), "wickets record not deleted");

        } catch (SQLException e) {
            fail("Exception occurred: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testMatchDeletion() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            // Insert dependencies
            insertDependencies("match", stmt);
            stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id) VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025')");

            // Insert related records
            stmt.execute("INSERT INTO awards (match_id, award_type, player_id) VALUES ('IPL2025001', 'orange_cap', 1)");
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, 1, 4, 'boundary')");
            stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', 1, 1, 1, 'wide', 1)");
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', 1, 1, 1, 1, 'bowled')");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 1, 'batter', 1, FALSE)");

            // Delete match record
            stmt.execute("DELETE FROM match WHERE match_id = 'IPL2025001'");

            // Verify related records are deleted
            ResultSet rs = stmt.executeQuery("SELECT * FROM awards WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "awards record not deleted");

            rs = stmt.executeQuery("SELECT * FROM balls WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "balls record not deleted");

            rs = stmt.executeQuery("SELECT * FROM batter_score WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "batter_score record not deleted");

            rs = stmt.executeQuery("SELECT * FROM extras WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "extras record not deleted");

            rs = stmt.executeQuery("SELECT * FROM wickets WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "wickets record not deleted");

            rs = stmt.executeQuery("SELECT * FROM player_match WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "player_match record not deleted");

        } catch (SQLException e) {
            fail("Exception occurred: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testSeasonDeletion() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();

            // Insert dependencies
            insertDependencies("awards", stmt);
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (2, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (3, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (1, 'IPL2025', 1, 1000000, 2000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (2, 'IPL2025', 2, 1000000, 2000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (3, 'IPL2025', 3, 1000000, 2000000, TRUE, 2)");

            // Insert related records
            // stmt.execute("INSERT INTO player_team (player_id, team_id, season_id) VALUES (1, 1, 'IPL2025')");
            stmt.execute("INSERT INTO awards (match_id, award_type, player_id) VALUES ('IPL2025001', 'orange_cap', 1)");
            stmt.execute("INSERT INTO player_match (match_id, player_id, role, team_id, is_extra) VALUES ('IPL2025001', 1, 'batter', 1, FALSE)");
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES ('IPL2025001', 1, 1, 1, 1, 2, 3)");
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES ('IPL2025001', 1, 1, 1, 4, 'boundary')");
            stmt.execute("INSERT INTO extras (match_id, innings_num, over_num, ball_num, extra_type, extra_runs) VALUES ('IPL2025001', 1, 1, 1, 'wide', 1)");
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES ('IPL2025001', 1, 1, 1, 1, 'bowled')");

            // Delete season record
            stmt.execute("DELETE FROM season WHERE season_id = 'IPL2025'");

            // Verify related records are deleted
            ResultSet rs = stmt.executeQuery("SELECT * FROM auction WHERE season_id = 'IPL2025'");
            assertFalse(rs.next(), "auction record not deleted");

            rs = stmt.executeQuery("SELECT * FROM player_team WHERE season_id = 'IPL2025'");
            assertFalse(rs.next(), "player_team record not deleted");

            rs = stmt.executeQuery("SELECT * FROM awards WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "awards record not deleted");

            rs = stmt.executeQuery("SELECT * FROM player_match WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "player_match record not deleted");

            rs = stmt.executeQuery("SELECT * FROM balls WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "balls record not deleted");

            rs = stmt.executeQuery("SELECT * FROM batter_score WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "batter_score record not deleted");

            rs = stmt.executeQuery("SELECT * FROM extras WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "extras record not deleted");

            rs = stmt.executeQuery("SELECT * FROM wickets WHERE match_id = 'IPL2025001'");
            assertFalse(rs.next(), "wickets record not deleted");

            rs = stmt.executeQuery("SELECT * FROM match WHERE season_id = 'IPL2025'");
            assertFalse(rs.next(), "match record not deleted");

        } catch (SQLException e) {
            fail("Exception occurred: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }

    @Test
    @Tag("3_mark")
    void testMatchUpdateAndAwards() throws Exception {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();
            insertDependencies("awards", stmt);
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (2, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES (3, 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (1, 'IPL2025', 1, 1000000, 2000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (2, 'IPL2025', 2, 1000000, 2000000, TRUE, 1)");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) VALUES (3, 'IPL2025', 3, 1000000, 2000000, TRUE, 2)");

            // Insert match with required NULL fields
            // stmt.execute("INSERT INTO match (match_id, match_type, venue, team_1_id, team_2_id, match_date, season_id, win_run_margin, win_by_wickets, win_type, toss_winner, toss_decide, winner_team_id) " +
            //              "VALUES ('IPL2025001', 'league', 'Test Region', 1, 2, '2025-05-01', 'IPL2025', NULL, NULL, NULL, NULL, NULL, NULL)");
    
            // Update match with toss details
            stmt.execute("UPDATE match SET toss_winner = 1, toss_decide = 'bat' WHERE match_id = 'IPL2025001'");
    
            // Insert balls first
            stmt.execute("INSERT INTO balls (match_id, innings_num, over_num, ball_num, striker_id, non_striker_id, bowler_id) VALUES " +
                         "('IPL2025001', 1, 1, 1, 1, 2, 3), " +
                         "('IPL2025001', 1, 1, 2, 1, 2, 3), " +
                         "('IPL2025001', 1, 1, 3, 1, 2, 3), " +
                         "('IPL2025001', 1, 1, 4, 1, 2, 3), " +
                         "('IPL2025001', 1, 1, 5, 1, 2, 3), " +
                         "('IPL2025001', 1, 1, 6, 1, 2, 3)");
    
            // Insert batter scores
            stmt.execute("INSERT INTO batter_score (match_id, innings_num, over_num, ball_num, run_scored, type_run) VALUES " +
                         "('IPL2025001', 1, 1, 1, 4, 'boundary'), " +
                         "('IPL2025001', 1, 1, 2, 6, 'boundary'), " +
                         "('IPL2025001', 1, 1, 3, 1, 'running'), " +
                         "('IPL2025001', 1, 1, 4, 2, 'running'), " +
                         "('IPL2025001', 1, 1, 5, 1, 'running'), " +
                         "('IPL2025001', 1, 1, 6, 1, 'running')");
    
            // Insert wicket
            stmt.execute("INSERT INTO wickets (match_id, innings_num, over_num, ball_num, player_out_id, kind_out) VALUES " +
                         "('IPL2025001', 1, 1, 6, 1, 'bowled')");
    
            // Update match with win details
            stmt.execute("UPDATE match SET win_run_margin = 10, win_type = 'runs', winner_team_id = 1 WHERE match_id = 'IPL2025001'");
    
            // Verify awards are inserted
            ResultSet rs = stmt.executeQuery("SELECT * FROM awards WHERE match_id = 'IPL2025001' AND award_type = 'orange_cap'");
            assertTrue(rs.next(), "Orange cap award not inserted");
    
            rs = stmt.executeQuery("SELECT * FROM awards WHERE match_id = 'IPL2025001' AND award_type = 'purple_cap'");
            assertTrue(rs.next(), "Purple cap award not inserted");
    
        } catch (SQLException e) {
            fail("Exception occurred: " + e.getMessage());
        } finally {
            connection.rollback(); // Rollback all changes
            connection.setAutoCommit(true); // Restore default behavior
        }
    }
}
