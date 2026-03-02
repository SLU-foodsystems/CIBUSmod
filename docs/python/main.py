def define_env(env):
    print("[macros] main.py loaded")
    # Shared Mermaid header (init directive + styles)
    MERMAID_INIT = r'''
%%{init: {
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 20,
    "padding": 3,
    "curve": "linear"
  },
  "themeVariables": {
    "fontSize": "12px"
  }
}}%%
'''
    MERMAID_STYLE = r'''
  %% -------------------------
  %% Styles
  %% -------------------------
  classDef mod_main fill:#509e2f80,stroke:#203f13,stroke-width:2px,font-size:13,color:#203f13;
  classDef mod_mgmt fill:#6ad1e380,stroke:#125560,stroke-width:2px,font-size:13,color:#125560;
  classDef mod_opt  fill:#d9d9d680,stroke:#43433e,stroke-width:2px,font-size:13,color:#43433e;

  classDef data  fill:#fee8c880,stroke:#b30000,stroke-width:0.5px,font-size:11,color:#b30000;
  classDef param fill:#d7f4ee80,stroke:#165044,stroke-width:0.5px,font-size:11,color:#165044;

  classDef method   fill:#ffffff80,stroke:#000000,stroke-width:1px,font-size:12,color:#000000;
  classDef helper   fill:#f5f5f580,stroke:#616161,stroke-width:2px,font-size:12,color:#212121;
  classDef settings fill:#ffd5f680,stroke:#aa0088,stroke-width:1px,font-size:11,color:#aa0088;
'''

    @env.macro
    def mermaid_init() -> str:
        """
        Emit Mermaid shared init
        """
        return MERMAID_INIT
    
    @env.macro
    def mermaid_style() -> str:
        """
        Emit a Mermaid shared classDefs.
        Usage:
          {{ mermaid_style() }}
        """
        return MERMAID_STYLE