from wsms.cli import main


def test_top_level_list(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "tunnel" in output
    assert "head" in output


def test_tunnel_list(capsys):
    assert main(["tunnel", "list", "--input", "0-2", "--output", "36-38"]) == 0
    output = capsys.readouterr().out
    assert "dense:0-2->36-38" in output


def test_qualified_head_list(capsys):
    assert main(["head", "list", "--source", "custom"]) == 0
    output = capsys.readouterr().out
    assert "custom:000000" in output



def test_info_reports_all_families(capsys):
    assert main(["info", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"released_tunnels"' in output
    assert '"dense"' in output
    assert '"sparse"' in output
    assert '"custom"' in output


def test_infer_help_is_available_without_loading_model(capsys):
    try:
        main(["infer", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "--target" in output
    assert "--head" in output
    assert "TOKEN=PATH" in output
