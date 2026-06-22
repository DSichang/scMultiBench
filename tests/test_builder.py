from multibench.run import registry, builder


def test_scalex_command():
    v = registry.get("SCALEX").select("diagonal", {"rna", "atac_gas"})
    cmd = builder.build_command(v, values={"rna": "a.h5", "atac_gas": "b.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/SCALEX/main_SCALEX.py",
                   "--path1", "a.h5", "--path2", "b.h5", "--save_path", "out/"]


def test_multigrate_paired_nargs_and_params():
    v = registry.get("Multigrate").select("vertical", {"rna", "adt"})
    cmd = builder.build_command(
        v, values={"rna": ["r1.h5", "r2.h5"], "adt": ["a1.h5"]}, out_dir="o/")
    # nargs roles expand; default params appended
    assert "--path1" in cmd and "r1.h5" in cmd and "r2.h5" in cmd
    assert "--epochs" in cmd and "200" in cmd
    assert "--bs" in cmd and "256" in cmd


def test_seurat_positional_R_command():
    v = registry.get("Seurat_v5").select("diagonal", {"rna", "atac_peak"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac_peak": "p.h5"}, out_dir="o/")
    assert cmd == ["Rscript", "tools_scripts/Seurat_v5/main_Seurat_v5.Rmd",
                   "r.h5", "r.h5", "p.h5", "p.h5", "o/"]


def test_scbridge_eq_and_labels():
    v = registry.get("scBridge").select("diagonal", {"rna", "atac_gas"})
    cmd = builder.build_command(v, values={
        "data_dir": "D27/", "source_data": "rna.h5", "target_data": "atac_gas.h5",
        "source_cty": "rna_cty.csv", "target_cty": "atac_cty.csv"}, out_dir="out/")
    assert "--data_path=D27/" in cmd
    assert "--source_data=rna.h5" in cmd
    assert "--source_cty" in cmd and "rna_cty.csv" in cmd
    assert "--save_path=out/" in cmd


def test_totalvi_command():
    v = registry.get("totalVI").select("vertical", {"rna", "adt"})
    cmd = builder.build_command(v, values={"rna": ["r.h5"], "adt": ["a.h5"]}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/totalVI/main_totalVI.py",
                   "--path1", "r.h5", "--path2", "a.h5", "--save_path", "out/"]


def test_multivi_command():
    v = registry.get("MultiVI").select("mosaic", {"rna", "atac", "rna_pair", "atac_pair"})
    cmd = builder.build_command(v, values={
        "rna": "r.h5", "atac": "a.h5", "rna_pair": "rp.h5", "atac_pair": "ap.h5"},
        out_dir="out/")
    assert cmd == ["python", "tools_scripts/MultiVI/main_MultiVI.py",
                   "--path1", "r.h5", "--path2", "a.h5",
                   "--pair_path1", "rp.h5", "--pair_path2", "ap.h5",
                   "--save_path", "out/"]


def test_cobolt_command():
    v = registry.get("Cobolt").select("mosaic", {"rna1", "rna2", "atac2", "atac3"})
    cmd = builder.build_command(v, values={
        "rna1": "r1.h5", "rna2": "r2.h5", "atac2": "a2.h5", "atac3": "a3.h5"},
        out_dir="out/")
    assert cmd == ["python", "tools_scripts/Cobolt/main_Cobolt.py",
                   "--path1", "r1.h5", "--path2", "r2.h5",
                   "--path3", "a2.h5", "--path4", "a3.h5",
                   "--save_path", "out/", "--batch_size", "128", "--lr", "0.005"]


def test_scmvp_command():
    v = registry.get("scMVP").select("vertical", {"rna", "atac"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac": "a.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/scMVP/main_scMVP.py",
                   "--path1", "r.h5", "--path2", "a.h5", "--save_path", "out/"]


def test_vimcca_rna_adt_command():
    v = registry.get("VIMCCA").select("vertical", {"rna", "adt"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "adt": "a.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/VIMCCA/main_VIMCCA_RNA_ADT.py",
                   "--path1", "r.h5", "--path2", "a.h5", "--save_path", "out/"]


def test_vimcca_rna_atac_command():
    v = registry.get("VIMCCA").select("vertical", {"rna", "atac"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac": "a.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/VIMCCA/main_VIMCCA_RNA_ATAC.py",
                   "--path1", "r.h5", "--path2", "a.h5", "--save_path", "out/"]


def test_uniport_command():
    v = registry.get("uniPort").select("diagonal", {"rna", "atac_gas"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac_gas": "g.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/uniPort/main_uniPort.py",
                   "--path1", "r.h5", "--path2", "g.h5", "--save_path", "out/",
                   "--seed", "1"]


def test_scican_command():
    v = registry.get("sciCAN").select("diagonal", {"rna", "atac_gas"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac_gas": "g.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/sciCAN/main_sciCAN.py",
                   "--path1", "r.h5", "--path2", "g.h5", "--save_path", "out/"]


def test_vipcca_command():
    v = registry.get("VIPCCA").select("diagonal", {"rna", "atac_gas"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac_gas": "g.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/VIPCCA/main_VIPCCA.py",
                   "--path1", "r.h5", "--path2", "g.h5", "--save_path", "out/"]


def test_portal_command():
    v = registry.get("Portal").select("diagonal", {"rna", "atac_gas"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac_gas": "g.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/Portal/main_Portal.py",
                   "--path1", "r.h5", "--path2", "g.h5", "--save_path", "out/"]


def test_smile_command():
    v = registry.get("SMILE").select("mosaic", {"rna_ref", "atac_ref", "rna_query", "atac_query"})
    cmd = builder.build_command(v, values={
        "rna_ref": "rr.h5", "atac_ref": "ar.h5",
        "rna_query": "rq.h5", "atac_query": "aq.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/SMILE/main_SMILE.py",
                   "--ref_path1", "rr.h5", "--ref_path2", "ar.h5",
                   "--query_path1", "rq.h5", "--query_path2", "aq.h5",
                   "--save_path", "out/"]


def test_multimap_command():
    v = registry.get("MultiMAP").select("diagonal", {"rna", "atac_peak", "atac_gas"})
    cmd = builder.build_command(v, values={
        "rna": "r.h5", "atac_peak": "p.h5", "atac_gas": "g.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/MultiMAP/main_MultiMAP.py",
                   "--path1", "r.h5", "--path2", "p.h5", "--path3", "g.h5",
                   "--save_path", "out/"]


def test_scmsi_command():
    v = registry.get("scMSI").select("vertical", {"rna", "adt"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "adt": "a.h5"}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/scMSI/main_scMSI.py",
                   "--path1", "r.h5", "--path2", "a.h5", "--save_path", "out/"]


def test_inmf_command():
    v = registry.get("iNMF").select("diagonal", {"rna", "atac_gas"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac_gas": "g.h5"}, out_dir="out/")
    assert cmd == ["Rscript", "tools_scripts/iNMF/main_iNMF.Rmd",
                   "r.h5", "g.h5", "out/"]


def test_conos_command():
    v = registry.get("Conos").select("diagonal", {"rna", "atac_gas"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac_gas": "g.h5"}, out_dir="out/")
    assert cmd == ["Rscript", "tools_scripts/Conos/main_Conos.Rmd",
                   "r.h5", "g.h5", "out/"]


def test_online_inmf_command():
    v = registry.get("online_iNMF").select("diagonal", {"rna", "atac_gas"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac_gas": "g.h5"}, out_dir="out/")
    assert cmd == ["Rscript", "tools_scripts/online iNMF/main_online_iNMF.Rmd",
                   "r.h5", "g.h5", "out/"]
    # the space-containing directory must remain a single argv element
    assert "tools_scripts/online iNMF/main_online_iNMF.Rmd" in cmd


def test_uinmf_vertical_command():
    v = registry.get("UINMF").select("vertical", {"rna", "adt"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "adt": "a.h5"}, out_dir="out/")
    assert cmd == ["Rscript", "tools_scripts/UINMF/main_UINMF_vertical.Rmd",
                   "r.h5", "a.h5", "out/"]


def test_uinmf_cross_command():
    v = registry.get("UINMF").select("cross", {"rna1", "rna2", "adt1", "adt2"})
    cmd = builder.build_command(v, values={
        "rna1": "r1.h5", "rna2": "r2.h5", "adt1": "a1.h5", "adt2": "a2.h5"},
        out_dir="out/")
    assert cmd == ["Rscript", "tools_scripts/UINMF/main_UINMF_cross.Rmd",
                   "r1.h5", "r2.h5", "a1.h5", "a2.h5", "out/"]


def test_scmomat_vertical_rna_adt_command():
    v = registry.get("scMoMaT").select("vertical", {"rna", "adt"})
    cmd = builder.build_command(v, values={"rna": ["r.h5"], "adt": ["a.h5"]}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/scMoMaT/main_scMoMaT.py",
                   "--path1", "r.h5", "--path2", "a.h5", "--save_path", "out/"]


def test_scmomat_vertical_rna_adt_atac_command():
    v = registry.get("scMoMaT").select("vertical", {"rna", "adt", "atac"})
    cmd = builder.build_command(
        v, values={"rna": ["r.h5"], "adt": ["a.h5"], "atac": ["t.h5"]}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/scMoMaT/main_scMoMaT.py",
                   "--path1", "r.h5", "--path2", "a.h5", "--path3", "t.h5",
                   "--save_path", "out/"]


def test_mira_command():
    v = registry.get("MIRA").select("vertical", {"rna", "atac"})
    cmd = builder.build_command(v, values={"rna": ["r.h5"], "atac": ["t.h5"]}, out_dir="out/")
    assert cmd == ["python", "tools_scripts/MIRA/main_MIRA.py",
                   "--rna", "r.h5", "--atac", "t.h5", "--save_path", "out/"]


def test_seurat_wnn_rna_adt_const_null():
    v = registry.get("Seurat_WNN").select("vertical", {"rna", "adt"})
    # atac is a const NULL placeholder; caller does NOT supply it
    cmd = builder.build_command(v, values={"rna": "r.h5", "adt": "a.h5"}, out_dir="out/")
    assert cmd == ["Rscript", "tools_scripts/Seurat_v4/main_Seurat_v4.Rmd",
                   "r.h5", "a.h5", "NULL", "out/"]


def test_seurat_wnn_rna_atac_const_null():
    v = registry.get("Seurat_WNN").select("vertical", {"rna", "atac"})
    cmd = builder.build_command(v, values={"rna": "r.h5", "atac": "t.h5"}, out_dir="out/")
    assert cmd == ["Rscript", "tools_scripts/Seurat_v4/main_Seurat_v4.Rmd",
                   "r.h5", "NULL", "t.h5", "out/"]
