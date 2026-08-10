class TestComparative:

    def test_t_test_different_means_significant(self, two_groups):
        from app.hypothesis_tests.comparative import t_test
        g = two_groups["groups"]
        r = t_test(g[0], g[1])
        assert r["test"] == "t_test"
        assert r["significant"] is True

    def test_t_test_identical_groups_not_significant(self, identical_groups):
        from app.hypothesis_tests.comparative import t_test
        g = identical_groups["groups"]
        r = t_test(g[0], g[1])
        assert r["significant"] is False

    def test_t_test_result_has_all_keys(self, two_groups):
        from app.hypothesis_tests.comparative import t_test
        g = two_groups["groups"]
        r = t_test(g[0], g[1])
        assert set(r.keys()) == {"test", "statistic", "p_value", "significant"}

    def test_t_test_significant_is_python_bool(self, two_groups):
        from app.hypothesis_tests.comparative import t_test
        g = two_groups["groups"]
        r = t_test(g[0], g[1])
        assert isinstance(r["significant"], bool)

    def test_mann_whitney_different_groups_significant(self, two_groups):
        from app.hypothesis_tests.comparative import mann_whitney_test
        g = two_groups["groups"]
        r = mann_whitney_test(g[0], g[1])
        assert r["test"] == "mann_whitney_test"
        assert r["significant"] is True

    def test_anova_three_groups_significant(self, three_groups):
        from app.hypothesis_tests.comparative import anova_test
        g = three_groups["groups"]
        r = anova_test(*g)
        assert r["test"] == "anova_test"
        assert r["significant"] is True
        assert r["f_statistic"] > 0

    def test_kruskal_wallis_three_groups_significant(self, three_groups):
        from app.hypothesis_tests.comparative import kruskal_wallis_test
        g = three_groups["groups"]
        r = kruskal_wallis_test(*g)
        assert r["test"] == "kruskal_wallis_test"
        assert r["significant"] is True
        assert r["h_statistic"] > 0

    def test_p_values_are_valid_probabilities(self, two_groups):
        from app.hypothesis_tests.comparative import t_test, mann_whitney_test
        g = two_groups["groups"]
        for fn in (t_test, mann_whitney_test):
            r = fn(g[0], g[1])
            assert 0 <= r["p_value"] <= 1