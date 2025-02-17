package com.example;

import org.junit.jupiter.api.*;
import java.sql.*;
import static org.junit.jupiter.api.Assertions.*;

public class PostgreSQLTest {
    private static Connection connection;
    private static final String DATABASE_NAME = "cricket_db";

    @BeforeAll
    static void setup() throws SQLException {
        connection = DatabaseConnection.getConnection(DATABASE_NAME);
    }

    @AfterAll
    static void tearDown() throws SQLException {
        if (connection != null) {
            connection.close();
        }
    }

    @BeforeEach
    void resetDatabase() throws SQLException {
        try (Statement stmt = connection.createStatement()) {
            stmt.execute("DELETE FROM auction");
            stmt.execute("DELETE FROM match");
            stmt.execute("DELETE FROM player");
            stmt.execute("DELETE FROM team");
            stmt.execute("DELETE FROM season");
        }
    }

    @Test
    @Tag("1_mark")
    void testPlayer_InvalidDOB() {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) " +
                         "VALUES ('P001', 'Test Player', '2017-01-01', 'right', 'India')");
            fail("Should throw exception for invalid DOB");
        } catch (SQLException e) {
            assertTrue(e.getMessage().toLowerCase().contains("check"), "Expected check constraint violation: " + e.getMessage());
        } finally {
            try {
                connection.rollback();
                connection.setAutoCommit(true);
            } catch (SQLException ignored) {}
        }
    }

    @Test
    @Tag("1_mark")
    void testAuction_InvalidSoldPrice() {
        try {
            connection.setAutoCommit(false);
            Statement stmt = connection.createStatement();
            stmt.execute("INSERT INTO season (season_id, year, start_date, end_date) VALUES ('S001', 2025, '2025-01-01', '2025-12-31')");
            stmt.execute("INSERT INTO player (player_id, player_name, dob, batting_hand, country_name) VALUES ('P001', 'Test Player', '2000-01-01', 'right', 'India')");
            stmt.execute("INSERT INTO team (team_id, team_name, coach_name, region) VALUES ('T001', 'Test Team', 'Coach Name', 'Region1')");
            stmt.execute("INSERT INTO auction (auction_id, season_id, player_id, base_price, sold_price, is_sold, team_id) " +
                         "VALUES ('A001', 'S001', 'P001', 2000000, 1500000, true, 'T001')");
            fail("Should throw exception for sold_price < base_price");
        } catch (SQLException e) {
            assertTrue(e.getMessage().toLowerCase().contains("check"), "Expected check constraint violation: " + e.getMessage());
        } finally {
            try {
                connection.rollback();
                connection.setAutoCommit(true);
            } catch (SQLException ignored) {}
        }
    }
}
