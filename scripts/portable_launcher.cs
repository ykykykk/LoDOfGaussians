using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;

class Launcher {
    [DllImport("kernel32.dll")] static extern IntPtr GetConsoleWindow();
    [DllImport("kernel32.dll")] static extern uint GetConsoleProcessList(uint[] ids, uint count);
    [DllImport("user32.dll")] static extern bool ShowWindow(IntPtr window, int command);
    static string Quote(string value) {
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in value) {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') { result.Append('\\', slashes * 2 + 1); result.Append(c); }
            else { result.Append('\\', slashes); result.Append(c); }
            slashes = 0;
        }
        result.Append('\\', slashes * 2); result.Append('"');
        return result.ToString();
    }
    static int Main(string[] args) {
        if (args.Length == 0 || (args.Length == 1 && args[0] == "ui")) {
            if (GetConsoleProcessList(new uint[2], 2) == 1) ShowWindow(GetConsoleWindow(), 0);
        }
        string root = AppDomain.CurrentDomain.BaseDirectory;
        var command = new StringBuilder("-I " + Quote(Path.Combine(root, "app", "portable_boot.py")));
        foreach (string arg in args) command.Append(" " + Quote(arg));
        var info = new ProcessStartInfo(Path.Combine(root, "runtime", "python.exe"), command.ToString());
        info.UseShellExecute = false;
        info.EnvironmentVariables.Remove("PYTHONHOME");
        info.EnvironmentVariables.Remove("PYTHONPATH");
        info.EnvironmentVariables["PYTHONUTF8"] = "1";
        try {
            using (var process = Process.Start(info)) {
                process.WaitForExit();
                return process.ExitCode;
            }
        } catch (Exception e) {
            Console.Error.WriteLine(e.Message);
            return 1;
        }
    }
}
