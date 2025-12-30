package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/types/model"
)

// MCPConfigFile represents the global MCP configuration file structure.
type MCPConfigFile struct {
	MCPServers map[string]MCPServerConfig `json:"mcpServers"`
}

// MCPServerConfig represents a single MCP server configuration.
type MCPServerConfig struct {
	Type     string            `json:"type,omitempty"`
	Command  string            `json:"command"`
	Args     []string          `json:"args,omitempty"`
	Env      map[string]string `json:"env,omitempty"`
	Disabled bool              `json:"disabled,omitempty"`
}

// getMCPConfigPath returns the path to the global MCP config file.
func getMCPConfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".ollama", "mcp.json")
}

// loadMCPConfig loads the global MCP configuration file.
func loadMCPConfig() (*MCPConfigFile, error) {
	configPath := getMCPConfigPath()
	if configPath == "" {
		return nil, fmt.Errorf("could not determine home directory")
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			// Return empty config if file doesn't exist
			return &MCPConfigFile{
				MCPServers: make(map[string]MCPServerConfig),
			}, nil
		}
		return nil, fmt.Errorf("reading config: %w", err)
	}

	var config MCPConfigFile
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	if config.MCPServers == nil {
		config.MCPServers = make(map[string]MCPServerConfig)
	}

	return &config, nil
}

// saveMCPConfig saves the global MCP configuration file.
func saveMCPConfig(config *MCPConfigFile) error {
	configPath := getMCPConfigPath()
	if configPath == "" {
		return fmt.Errorf("could not determine home directory")
	}

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return fmt.Errorf("creating config directory: %w", err)
	}

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}

	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		return fmt.Errorf("writing config: %w", err)
	}

	return nil
}

// MCPAddHandler handles the mcp add command.
func MCPAddHandler(cmd *cobra.Command, args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: ollama mcp add NAME COMMAND [ARGS...]")
	}

	name := args[0]
	command := args[1]
	cmdArgs := args[2:]

	// Load existing config
	config, err := loadMCPConfig()
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	// Check if already exists
	if _, exists := config.MCPServers[name]; exists {
		fmt.Fprintf(os.Stderr, "Warning: overwriting existing MCP server '%s'\n", name)
	}

	// Add the new server
	config.MCPServers[name] = MCPServerConfig{
		Type:    "stdio",
		Command: command,
		Args:    cmdArgs,
	}

	// Save config
	if err := saveMCPConfig(config); err != nil {
		return fmt.Errorf("saving config: %w", err)
	}

	configPath := getMCPConfigPath()
	fmt.Fprintf(os.Stderr, "Added MCP server '%s' to %s\n", name, configPath)
	fmt.Fprintf(os.Stderr, "  Command: %s %s\n", command, strings.Join(cmdArgs, " "))

	return nil
}

// MCPRemoveGlobalHandler handles removing an MCP from global config.
func MCPRemoveGlobalHandler(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: ollama mcp remove-global NAME [NAME...]")
	}

	config, err := loadMCPConfig()
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	for _, name := range args {
		if _, exists := config.MCPServers[name]; !exists {
			fmt.Fprintf(os.Stderr, "MCP server '%s' not found in global config\n", name)
			continue
		}

		delete(config.MCPServers, name)
		fmt.Fprintf(os.Stderr, "Removed MCP server '%s' from global config\n", name)
	}

	if err := saveMCPConfig(config); err != nil {
		return fmt.Errorf("saving config: %w", err)
	}

	return nil
}

// MCPListGlobalHandler handles listing global MCP servers.
func MCPListGlobalHandler(cmd *cobra.Command, args []string) error {
	config, err := loadMCPConfig()
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	if len(config.MCPServers) == 0 {
		fmt.Println("No global MCP servers configured")
		fmt.Printf("Add one with: ollama mcp add NAME COMMAND [ARGS...]\n")
		return nil
	}

	fmt.Printf("Global MCP servers (%s):\n\n", getMCPConfigPath())

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "NAME\tCOMMAND\tSTATUS")

	for name, srv := range config.MCPServers {
		cmdLine := srv.Command
		if len(srv.Args) > 0 {
			cmdLine += " " + strings.Join(srv.Args, " ")
		}
		status := "enabled"
		if srv.Disabled {
			status = "disabled"
		}
		fmt.Fprintf(w, "%s\t%s\t%s\n", name, cmdLine, status)
	}

	return w.Flush()
}

// MCPDisableHandler handles disabling an MCP server in global config.
func MCPDisableHandler(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: ollama mcp disable NAME [NAME...]")
	}

	config, err := loadMCPConfig()
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	for _, name := range args {
		srv, exists := config.MCPServers[name]
		if !exists {
			fmt.Fprintf(os.Stderr, "MCP server '%s' not found in global config\n", name)
			continue
		}

		if srv.Disabled {
			fmt.Fprintf(os.Stderr, "MCP server '%s' is already disabled\n", name)
			continue
		}

		srv.Disabled = true
		config.MCPServers[name] = srv
		fmt.Fprintf(os.Stderr, "Disabled MCP server '%s'\n", name)
	}

	if err := saveMCPConfig(config); err != nil {
		return fmt.Errorf("saving config: %w", err)
	}

	return nil
}

// MCPEnableHandler handles enabling an MCP server in global config.
func MCPEnableHandler(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: ollama mcp enable NAME [NAME...]")
	}

	config, err := loadMCPConfig()
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	for _, name := range args {
		srv, exists := config.MCPServers[name]
		if !exists {
			fmt.Fprintf(os.Stderr, "MCP server '%s' not found in global config\n", name)
			continue
		}

		if !srv.Disabled {
			fmt.Fprintf(os.Stderr, "MCP server '%s' is already enabled\n", name)
			continue
		}

		srv.Disabled = false
		config.MCPServers[name] = srv
		fmt.Fprintf(os.Stderr, "Enabled MCP server '%s'\n", name)
	}

	if err := saveMCPConfig(config); err != nil {
		return fmt.Errorf("saving config: %w", err)
	}

	return nil
}

// MCPPushHandler handles the mcp push command.
func MCPPushHandler(cmd *cobra.Command, args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: ollama mcp push NAME[:TAG] PATH")
	}

	name := args[0]
	path := args[1]

	// Expand path
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("expanding home directory: %w", err)
		}
		path = filepath.Join(home, path[1:])
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("resolving path: %w", err)
	}

	// Validate MCP directory - check for mcp.json, package.json, or any config file
	validFiles := []string{"mcp.json", "package.json", "server.py", "server.js", "main.py", "index.js"}
	found := false
	for _, vf := range validFiles {
		if _, err := os.Stat(filepath.Join(absPath, vf)); err == nil {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("MCP directory should contain one of: %s", strings.Join(validFiles, ", "))
	}

	// Parse MCP name (will set Kind="mcp")
	n := server.ParseMCPName(name)
	if n.Model == "" {
		return fmt.Errorf("invalid MCP name: %s", name)
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	// Create MCP layer
	displayName := n.DisplayShortest()
	status := fmt.Sprintf("Creating MCP layer for %s", displayName)
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)

	layer, err := server.CreateMCPLayer(absPath)
	if err != nil {
		return fmt.Errorf("creating MCP layer: %w", err)
	}

	spinner.Stop()

	// Create MCP manifest
	manifest, configLayer, err := createMCPManifest(absPath, layer)
	if err != nil {
		return fmt.Errorf("creating MCP manifest: %w", err)
	}

	// Write manifest locally
	manifestPath, err := server.GetMCPManifestPath(n)
	if err != nil {
		return fmt.Errorf("getting manifest path: %w", err)
	}

	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		return fmt.Errorf("creating manifest directory: %w", err)
	}

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return fmt.Errorf("marshaling manifest: %w", err)
	}

	if err := os.WriteFile(manifestPath, manifestJSON, 0o644); err != nil {
		return fmt.Errorf("writing manifest: %w", err)
	}

	fmt.Fprintf(os.Stderr, "MCP %s created locally\n", displayName)
	fmt.Fprintf(os.Stderr, "  Config: %s (%s)\n", configLayer.Digest, format.HumanBytes(configLayer.Size))
	fmt.Fprintf(os.Stderr, "  Layer:  %s (%s)\n", layer.Digest, format.HumanBytes(layer.Size))

	// Push to registry
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("creating client: %w", err)
	}

	insecure, _ := cmd.Flags().GetBool("insecure")

	fmt.Fprintf(os.Stderr, "\nPushing to registry...\n")

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			bar := progress.NewBar(resp.Status, resp.Total, resp.Completed)
			p.Add(resp.Digest, bar)
		} else if resp.Status != "" {
			spinner := progress.NewSpinner(resp.Status)
			p.Add(resp.Status, spinner)
		}
		return nil
	}

	req := &api.PushRequest{
		Model:    displayName,
		Insecure: insecure,
	}

	if err := client.Push(context.Background(), req, fn); err != nil {
		// If push fails, still show success for local creation
		fmt.Fprintf(os.Stderr, "\nNote: Local MCP created but push failed: %v\n", err)
		fmt.Fprintf(os.Stderr, "You can try pushing later with: ollama mcp push %s\n", name)
		return nil
	}

	fmt.Fprintf(os.Stderr, "Successfully pushed %s\n", displayName)
	return nil
}

// MCPPullHandler handles the mcp pull command.
func MCPPullHandler(cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: ollama mcp pull NAME[:TAG]")
	}

	name := args[0]
	n := server.ParseMCPName(name)
	if n.Model == "" {
		return fmt.Errorf("invalid MCP name: %s", name)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("creating client: %w", err)
	}

	insecure, _ := cmd.Flags().GetBool("insecure")

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			bar := progress.NewBar(resp.Status, resp.Total, resp.Completed)
			p.Add(resp.Digest, bar)
		} else if resp.Status != "" {
			spinner := progress.NewSpinner(resp.Status)
			p.Add(resp.Status, spinner)
		}
		return nil
	}

	displayName := n.DisplayShortest()
	req := &api.PullRequest{
		Model:    displayName,
		Insecure: insecure,
	}

	if err := client.Pull(context.Background(), req, fn); err != nil {
		return fmt.Errorf("pulling MCP: %w", err)
	}

	fmt.Fprintf(os.Stderr, "Successfully pulled %s\n", displayName)
	return nil
}

// MCPListHandler handles the mcp list command.
func MCPListHandler(cmd *cobra.Command, args []string) error {
	mcps, err := listLocalMCPs()
	if err != nil {
		return fmt.Errorf("listing MCPs: %w", err)
	}

	if len(mcps) == 0 {
		fmt.Println("No MCPs installed")
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "NAME\tTAG\tSIZE\tMODIFIED")

	for _, mcp := range mcps {
		fmt.Fprintf(w, "%s/%s\t%s\t%s\t%s\n",
			mcp.Namespace,
			mcp.Name,
			mcp.Tag,
			format.HumanBytes(mcp.Size),
			format.HumanTime(mcp.ModifiedAt, "Never"),
		)
	}

	return w.Flush()
}

// MCPRemoveHandler handles the mcp rm command.
func MCPRemoveHandler(cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: ollama mcp rm NAME[:TAG] [NAME[:TAG]...]")
	}

	for _, name := range args {
		n := server.ParseMCPName(name)
		if n.Model == "" {
			fmt.Fprintf(os.Stderr, "Invalid MCP name: %s\n", name)
			continue
		}

		displayName := n.DisplayShortest()
		manifestPath, err := server.GetMCPManifestPath(n)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting manifest path for %s: %v\n", name, err)
			continue
		}

		if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "MCP not found: %s\n", displayName)
			continue
		}

		if err := os.Remove(manifestPath); err != nil {
			fmt.Fprintf(os.Stderr, "Error removing %s: %v\n", displayName, err)
			continue
		}

		// Clean up empty parent directories
		dir := filepath.Dir(manifestPath)
		for dir != filepath.Join(os.Getenv("HOME"), ".ollama", "models", "manifests") {
			entries, _ := os.ReadDir(dir)
			if len(entries) == 0 {
				os.Remove(dir)
				dir = filepath.Dir(dir)
			} else {
				break
			}
		}

		fmt.Fprintf(os.Stderr, "Deleted '%s'\n", displayName)
	}

	return nil
}

// MCPShowHandler handles the mcp show command.
func MCPShowHandler(cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: ollama mcp show NAME[:TAG]")
	}

	name := args[0]
	n := server.ParseMCPName(name)
	if n.Model == "" {
		return fmt.Errorf("invalid MCP name: %s", name)
	}

	displayName := n.DisplayShortest()
	manifestPath, err := server.GetMCPManifestPath(n)
	if err != nil {
		return fmt.Errorf("getting manifest path: %w", err)
	}

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("MCP not found: %s", displayName)
		}
		return fmt.Errorf("reading manifest: %w", err)
	}

	var manifest server.Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("parsing manifest: %w", err)
	}

	fmt.Printf("MCP: %s\n\n", displayName)

	fmt.Println("Layers:")
	for _, layer := range manifest.Layers {
		fmt.Printf("  %s  %s  %s\n", layer.MediaType, layer.Digest[:19], format.HumanBytes(layer.Size))
	}

	// Try to read and display mcp.json or package.json content
	if len(manifest.Layers) > 0 {
		for _, layer := range manifest.Layers {
			if layer.MediaType == server.MediaTypeMCP {
				mcpPath, err := server.GetMCPsPath(layer.Digest)
				if err == nil {
					// Try mcp.json first
					mcpJSONPath := filepath.Join(mcpPath, "mcp.json")
					if content, err := os.ReadFile(mcpJSONPath); err == nil {
						fmt.Println("\nConfig (mcp.json):")
						fmt.Println(string(content))
					} else {
						// Try package.json
						pkgJSONPath := filepath.Join(mcpPath, "package.json")
						if content, err := os.ReadFile(pkgJSONPath); err == nil {
							fmt.Println("\nConfig (package.json):")
							fmt.Println(string(content))
						}
					}

					// List files in the MCP
					fmt.Println("\nFiles:")
					filepath.Walk(mcpPath, func(path string, info os.FileInfo, err error) error {
						if err != nil {
							return nil
						}
						relPath, _ := filepath.Rel(mcpPath, path)
						if relPath == "." {
							return nil
						}
						if info.IsDir() {
							fmt.Printf("  %s/\n", relPath)
						} else {
							fmt.Printf("  %s (%s)\n", relPath, format.HumanBytes(info.Size()))
						}
						return nil
					})
				}
			}
		}
	}

	return nil
}

// MCPInfo represents information about an installed MCP.
type MCPInfo struct {
	Namespace  string
	Name       string
	Tag        string
	Size       int64
	ModifiedAt time.Time
}

// listLocalMCPs returns a list of locally installed MCPs.
// MCPs are stored with 5-part paths: host/namespace/kind/model/tag
// where kind is "mcp".
func listLocalMCPs() ([]MCPInfo, error) {
	manifestsPath := filepath.Join(os.Getenv("HOME"), ".ollama", "models", "manifests")

	var mcps []MCPInfo

	// Walk through all registries
	registries, err := os.ReadDir(manifestsPath)
	if err != nil {
		if os.IsNotExist(err) {
			return mcps, nil
		}
		return nil, err
	}

	for _, registry := range registries {
		if !registry.IsDir() {
			continue
		}

		// Walk namespaces
		namespaces, err := os.ReadDir(filepath.Join(manifestsPath, registry.Name()))
		if err != nil {
			continue
		}

		for _, namespace := range namespaces {
			if !namespace.IsDir() {
				continue
			}

			// Walk kinds looking for "mcp"
			kinds, err := os.ReadDir(filepath.Join(manifestsPath, registry.Name(), namespace.Name()))
			if err != nil {
				continue
			}

			for _, kind := range kinds {
				if !kind.IsDir() {
					continue
				}

				// Only process mcp kind
				if kind.Name() != server.MCPNamespace {
					continue
				}

				// Walk MCP names (model names)
				mcpNames, err := os.ReadDir(filepath.Join(manifestsPath, registry.Name(), namespace.Name(), kind.Name()))
				if err != nil {
					continue
				}

				for _, mcpName := range mcpNames {
					if !mcpName.IsDir() {
						continue
					}

					// Walk tags
					tags, err := os.ReadDir(filepath.Join(manifestsPath, registry.Name(), namespace.Name(), kind.Name(), mcpName.Name()))
					if err != nil {
						continue
					}

					for _, tag := range tags {
						manifestPath := filepath.Join(manifestsPath, registry.Name(), namespace.Name(), kind.Name(), mcpName.Name(), tag.Name())
						fi, err := os.Stat(manifestPath)
						if err != nil || fi.IsDir() {
							continue
						}

						// Read manifest to get size
						data, err := os.ReadFile(manifestPath)
						if err != nil {
							continue
						}

						var manifest server.Manifest
						if err := json.Unmarshal(data, &manifest); err != nil {
							continue
						}

						var totalSize int64
						for _, layer := range manifest.Layers {
							totalSize += layer.Size
						}

						// Build display name using model.Name
						n := model.Name{
							Host:      registry.Name(),
							Namespace: namespace.Name(),
							Kind:      kind.Name(),
							Model:     mcpName.Name(),
							Tag:       tag.Name(),
						}

						mcps = append(mcps, MCPInfo{
							Namespace:  n.Namespace + "/" + n.Kind,
							Name:       n.Model,
							Tag:        n.Tag,
							Size:       totalSize,
							ModifiedAt: fi.ModTime(),
						})
					}
				}
			}
		}
	}

	return mcps, nil
}

// createMCPManifest creates a manifest for a standalone MCP.
func createMCPManifest(mcpDir string, layer server.Layer) (*server.Manifest, *server.Layer, error) {
	// Try to read mcp.json or package.json to extract metadata
	name, description := extractMCPMetadata(mcpDir)
	if name == "" {
		// Use directory name as fallback
		name = filepath.Base(mcpDir)
	}

	// Create config
	config := map[string]any{
		"name":         name,
		"description":  description,
		"architecture": "amd64",
		"os":           "linux",
	}

	configJSON, err := json.Marshal(config)
	if err != nil {
		return nil, nil, fmt.Errorf("marshaling config: %w", err)
	}

	// Create config layer
	configLayer, err := server.NewLayer(strings.NewReader(string(configJSON)), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return nil, nil, fmt.Errorf("creating config layer: %w", err)
	}

	manifest := &server.Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config:        configLayer,
		Layers:        []server.Layer{layer},
	}

	return manifest, &configLayer, nil
}

// extractMCPMetadata extracts name and description from mcp.json or package.json.
func extractMCPMetadata(mcpDir string) (name, description string) {
	// Try mcp.json first
	mcpJSONPath := filepath.Join(mcpDir, "mcp.json")
	if data, err := os.ReadFile(mcpJSONPath); err == nil {
		var config map[string]any
		if err := json.Unmarshal(data, &config); err == nil {
			if n, ok := config["name"].(string); ok {
				name = n
			}
			if d, ok := config["description"].(string); ok {
				description = d
			}
			return name, description
		}
	}

	// Try package.json
	pkgJSONPath := filepath.Join(mcpDir, "package.json")
	if data, err := os.ReadFile(pkgJSONPath); err == nil {
		var config map[string]any
		if err := json.Unmarshal(data, &config); err == nil {
			if n, ok := config["name"].(string); ok {
				name = n
			}
			if d, ok := config["description"].(string); ok {
				description = d
			}
			return name, description
		}
	}

	return "", ""
}

// NewMCPCommand creates the mcp parent command with subcommands.
func NewMCPCommand() *cobra.Command {
	mcpCmd := &cobra.Command{
		Use:   "mcp",
		Short: "Manage MCP servers",
		Long:  "Commands for managing MCP (Model Context Protocol) servers (add, push, pull, list, rm, show)",
	}

	// Global config commands
	addCmd := &cobra.Command{
		Use:   "add NAME COMMAND [ARGS...]",
		Short: "Add an MCP server to global config",
		Long: `Add an MCP server to the global config (~/.ollama/mcp.json).
Global MCP servers are available to all agents.

Examples:
  ollama mcp add web-search uv run ./mcp-server.py
  ollama mcp add calculator python3 /path/to/calc.py`,
		Args:               cobra.MinimumNArgs(2),
		RunE:               MCPAddHandler,
		DisableFlagParsing: true, // Allow args with dashes
	}

	removeGlobalCmd := &cobra.Command{
		Use:     "remove-global NAME [NAME...]",
		Aliases: []string{"rm-global"},
		Short:   "Remove an MCP server from global config",
		Args:    cobra.MinimumNArgs(1),
		RunE:    MCPRemoveGlobalHandler,
	}

	listGlobalCmd := &cobra.Command{
		Use:   "list-global",
		Short: "List global MCP servers",
		Args:  cobra.NoArgs,
		RunE:  MCPListGlobalHandler,
	}

	// Registry commands
	pushCmd := &cobra.Command{
		Use:     "push NAME[:TAG] PATH",
		Short:   "Push an MCP server to a registry",
		Long:    "Package a local MCP server directory and push it to a registry",
		Args:    cobra.ExactArgs(2),
		PreRunE: checkServerHeartbeat,
		RunE:    MCPPushHandler,
	}
	pushCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	pullCmd := &cobra.Command{
		Use:     "pull NAME[:TAG]",
		Short:   "Pull an MCP server from a registry",
		Args:    cobra.ExactArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE:    MCPPullHandler,
	}
	pullCmd.Flags().Bool("insecure", false, "Use an insecure registry")

	listCmd := &cobra.Command{
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List installed MCP servers (from registry)",
		Args:    cobra.NoArgs,
		RunE:    MCPListHandler,
	}

	rmCmd := &cobra.Command{
		Use:     "rm NAME[:TAG] [NAME[:TAG]...]",
		Aliases: []string{"remove", "delete"},
		Short:   "Remove an MCP server (from registry)",
		Args:    cobra.MinimumNArgs(1),
		RunE:    MCPRemoveHandler,
	}

	showCmd := &cobra.Command{
		Use:   "show NAME[:TAG]",
		Short: "Show MCP server details",
		Args:  cobra.ExactArgs(1),
		RunE:  MCPShowHandler,
	}

	disableCmd := &cobra.Command{
		Use:   "disable NAME [NAME...]",
		Short: "Disable an MCP server (keep in config)",
		Long: `Disable an MCP server without removing it from config.
Disabled servers will not be started when running agents.
Use 'ollama mcp enable' to re-enable.`,
		Args: cobra.MinimumNArgs(1),
		RunE: MCPDisableHandler,
	}

	enableCmd := &cobra.Command{
		Use:   "enable NAME [NAME...]",
		Short: "Enable a disabled MCP server",
		Long:  `Re-enable a previously disabled MCP server.`,
		Args:  cobra.MinimumNArgs(1),
		RunE:  MCPEnableHandler,
	}

	mcpCmd.AddCommand(addCmd, removeGlobalCmd, listGlobalCmd, disableCmd, enableCmd, pushCmd, pullCmd, listCmd, rmCmd, showCmd)

	return mcpCmd
}
