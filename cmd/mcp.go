package cmd

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	mcpInitTimeout     = 30 * time.Second
	mcpCallTimeout     = 60 * time.Second
	mcpShutdownTimeout = 5 * time.Second
)

// JSON-RPC types
type jsonrpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int    `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type jsonrpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonrpcError   `json:"error,omitempty"`
}

type jsonrpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// MCP protocol types
type mcpInitializeParams struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ClientInfo      mcpClientInfo  `json:"clientInfo"`
}

type mcpClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type mcpInitializeResult struct {
	ProtocolVersion string          `json:"protocolVersion"`
	Capabilities    mcpCapabilities `json:"capabilities"`
	ServerInfo      mcpServerInfo   `json:"serverInfo"`
}

type mcpCapabilities struct {
	Tools *mcpToolsCapability `json:"tools,omitempty"`
}

type mcpToolsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

type mcpServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type mcpTool struct {
	Name        string             `json:"name"`
	Description string             `json:"description,omitempty"`
	InputSchema mcpToolInputSchema `json:"inputSchema"`
}

type mcpToolInputSchema struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties,omitempty"`
	Required   []string       `json:"required,omitempty"`
}

type mcpToolsListResult struct {
	Tools []mcpTool `json:"tools"`
}

type mcpToolCallParams struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments,omitempty"`
}

type mcpToolCallResult struct {
	Content []mcpContent `json:"content"`
	IsError bool         `json:"isError,omitempty"`
}

type mcpContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// mcpServer represents a running MCP server process
type mcpServer struct {
	ref     api.MCPRef
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  *bufio.Reader
	stderr  io.ReadCloser
	tools   []mcpTool
	mu      sync.Mutex
	nextID  int
	started bool
}

// mcpManager manages multiple MCP servers for an agent session
type mcpManager struct {
	servers map[string]*mcpServer
	mu      sync.RWMutex
}

// newMCPManager creates a new MCP manager
func newMCPManager() *mcpManager {
	return &mcpManager{
		servers: make(map[string]*mcpServer),
	}
}

// loadMCPsFromRefs initializes MCP servers from refs
func (m *mcpManager) loadMCPsFromRefs(refs []api.MCPRef) error {
	if len(refs) == 0 {
		return nil
	}

	for _, ref := range refs {
		if err := m.addServer(ref); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to initialize MCP server %q: %v\n", ref.Name, err)
		}
	}

	return nil
}

// addServer adds and starts an MCP server
func (m *mcpManager) addServer(ref api.MCPRef) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.servers[ref.Name]; exists {
		return fmt.Errorf("MCP server %q already exists", ref.Name)
	}

	srv := &mcpServer{
		ref:    ref,
		nextID: 1,
	}

	if err := srv.start(); err != nil {
		return fmt.Errorf("starting MCP server: %w", err)
	}

	m.servers[ref.Name] = srv
	return nil
}

// start starts the MCP server process
func (s *mcpServer) start() error {
	s.mu.Lock()

	if s.started {
		s.mu.Unlock()
		return nil
	}

	s.cmd = exec.Command(s.ref.Command, s.ref.Args...)

	// Set environment
	s.cmd.Env = os.Environ()
	for k, v := range s.ref.Env {
		s.cmd.Env = append(s.cmd.Env, fmt.Sprintf("%s=%s", k, v))
	}

	var err error
	s.stdin, err = s.cmd.StdinPipe()
	if err != nil {
		s.mu.Unlock()
		return fmt.Errorf("creating stdin pipe: %w", err)
	}

	stdout, err := s.cmd.StdoutPipe()
	if err != nil {
		s.mu.Unlock()
		return fmt.Errorf("creating stdout pipe: %w", err)
	}
	s.stdout = bufio.NewReader(stdout)

	s.stderr, err = s.cmd.StderrPipe()
	if err != nil {
		s.mu.Unlock()
		return fmt.Errorf("creating stderr pipe: %w", err)
	}

	// Start stderr reader goroutine (discard stderr for now)
	go func() {
		scanner := bufio.NewScanner(s.stderr)
		for scanner.Scan() {
			_ = scanner.Text()
		}
	}()

	if err := s.cmd.Start(); err != nil {
		s.mu.Unlock()
		return fmt.Errorf("starting process: %w", err)
	}

	s.started = true
	s.mu.Unlock() // Release lock before calling initialize/listTools which use the mutex

	// Initialize the server
	if err := s.initialize(); err != nil {
		s.stop()
		return fmt.Errorf("initializing MCP server: %w", err)
	}

	// Get available tools
	if err := s.listTools(); err != nil {
		s.stop()
		return fmt.Errorf("listing tools: %w", err)
	}

	return nil
}

// initialize sends the MCP initialize request
func (s *mcpServer) initialize() error {
	ctx, cancel := context.WithTimeout(context.Background(), mcpInitTimeout)
	defer cancel()

	params := mcpInitializeParams{
		ProtocolVersion: "2024-11-05",
		Capabilities:    map[string]any{},
		ClientInfo: mcpClientInfo{
			Name:    "ollama",
			Version: "0.1.0",
		},
	}

	var result mcpInitializeResult
	if err := s.call(ctx, "initialize", params, &result); err != nil {
		return err
	}

	// Send initialized notification
	return s.notify("notifications/initialized", nil)
}

// listTools fetches the available tools from the MCP server
func (s *mcpServer) listTools() error {
	ctx, cancel := context.WithTimeout(context.Background(), mcpInitTimeout)
	defer cancel()

	var result mcpToolsListResult
	if err := s.call(ctx, "tools/list", nil, &result); err != nil {
		return err
	}

	s.tools = result.Tools
	return nil
}

// call sends a JSON-RPC request and waits for the response
func (s *mcpServer) call(ctx context.Context, method string, params any, result any) error {
	s.mu.Lock()
	id := s.nextID
	s.nextID++
	s.mu.Unlock()

	req := jsonrpcRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	reqBytes, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshaling request: %w", err)
	}

	// Send request
	s.mu.Lock()
	_, err = s.stdin.Write(append(reqBytes, '\n'))
	s.mu.Unlock()
	if err != nil {
		return fmt.Errorf("writing request: %w", err)
	}

	// Read response with timeout
	respCh := make(chan []byte, 1)
	errCh := make(chan error, 1)

	go func() {
		s.mu.Lock()
		line, err := s.stdout.ReadBytes('\n')
		s.mu.Unlock()
		if err != nil {
			errCh <- err
			return
		}
		respCh <- line
	}()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-errCh:
		return fmt.Errorf("reading response: %w", err)
	case line := <-respCh:
		var resp jsonrpcResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			return fmt.Errorf("unmarshaling response: %w", err)
		}

		if resp.Error != nil {
			return fmt.Errorf("MCP error %d: %s", resp.Error.Code, resp.Error.Message)
		}

		if result != nil && len(resp.Result) > 0 {
			if err := json.Unmarshal(resp.Result, result); err != nil {
				return fmt.Errorf("unmarshaling result: %w", err)
			}
		}

		return nil
	}
}

// notify sends a JSON-RPC notification (no response expected)
func (s *mcpServer) notify(method string, params any) error {
	req := jsonrpcRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}

	reqBytes, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshaling notification: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.stdin.Write(append(reqBytes, '\n')); err != nil {
		return fmt.Errorf("writing notification: %w", err)
	}

	return nil
}

// callTool executes a tool call on the MCP server
func (s *mcpServer) callTool(ctx context.Context, name string, arguments map[string]any) (string, error) {
	params := mcpToolCallParams{
		Name:      name,
		Arguments: arguments,
	}

	var result mcpToolCallResult
	if err := s.call(ctx, "tools/call", params, &result); err != nil {
		return "", err
	}

	// Concatenate text content
	var sb strings.Builder
	for _, content := range result.Content {
		if content.Type == "text" {
			sb.WriteString(content.Text)
		}
	}

	if result.IsError {
		return sb.String(), errors.New(sb.String())
	}

	return sb.String(), nil
}

// stop shuts down the MCP server
func (s *mcpServer) stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.started {
		return nil
	}

	// Close stdin to signal shutdown
	if s.stdin != nil {
		s.stdin.Close()
	}

	// Wait for process with timeout
	done := make(chan error, 1)
	go func() {
		done <- s.cmd.Wait()
	}()

	select {
	case <-time.After(mcpShutdownTimeout):
		s.cmd.Process.Kill()
	case <-done:
	}

	s.started = false
	return nil
}

// Tools returns all tools from all MCP servers as api.Tools
func (m *mcpManager) Tools() api.Tools {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var tools api.Tools

	for serverName, srv := range m.servers {
		for _, t := range srv.tools {
			// Namespace tool names: mcp_{servername}_{toolname}
			namespacedName := fmt.Sprintf("mcp_%s_%s", serverName, t.Name)

			tool := api.Tool{
				Type: "function",
				Function: api.ToolFunction{
					Name:        namespacedName,
					Description: t.Description,
					Parameters:  convertMCPSchema(t.InputSchema),
				},
			}
			tools = append(tools, tool)
		}
	}

	return tools
}

// convertMCPSchema converts MCP input schema to api.ToolFunctionParameters
func convertMCPSchema(schema mcpToolInputSchema) api.ToolFunctionParameters {
	params := api.ToolFunctionParameters{
		Type:       schema.Type,
		Required:   schema.Required,
		Properties: make(map[string]api.ToolProperty),
	}

	for name, prop := range schema.Properties {
		if propMap, ok := prop.(map[string]any); ok {
			tp := api.ToolProperty{}
			if t, ok := propMap["type"].(string); ok {
				tp.Type = api.PropertyType{t}
			}
			if d, ok := propMap["description"].(string); ok {
				tp.Description = d
			}
			params.Properties[name] = tp
		}
	}

	return params
}

// RunToolCall routes a tool call to the appropriate MCP server
func (m *mcpManager) RunToolCall(call api.ToolCall) (api.Message, bool, error) {
	name := call.Function.Name

	// Check if this is an MCP tool (mcp_servername_toolname)
	if !strings.HasPrefix(name, "mcp_") {
		return api.Message{}, false, nil
	}

	// Parse server name and tool name
	rest := strings.TrimPrefix(name, "mcp_")
	idx := strings.Index(rest, "_")
	if idx == -1 {
		return toolMessage(call, fmt.Sprintf("invalid MCP tool name: %s", name)), true, nil
	}

	serverName := rest[:idx]
	toolName := rest[idx+1:]

	m.mu.RLock()
	srv, ok := m.servers[serverName]
	m.mu.RUnlock()

	if !ok {
		return toolMessage(call, fmt.Sprintf("MCP server %q not found", serverName)), true, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), mcpCallTimeout)
	defer cancel()

	result, err := srv.callTool(ctx, toolName, call.Function.Arguments)
	if err != nil {
		return toolMessage(call, fmt.Sprintf("error: %v", err)), true, nil
	}

	return toolMessage(call, result), true, nil
}

// Shutdown stops all MCP servers
func (m *mcpManager) Shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, srv := range m.servers {
		srv.stop()
	}

	m.servers = make(map[string]*mcpServer)
}

// ServerNames returns the names of all running MCP servers
func (m *mcpManager) ServerNames() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	names := make([]string, 0, len(m.servers))
	for name := range m.servers {
		names = append(names, name)
	}
	return names
}

// ToolCount returns the total number of tools across all servers
func (m *mcpManager) ToolCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	count := 0
	for _, srv := range m.servers {
		count += len(srv.tools)
	}
	return count
}
